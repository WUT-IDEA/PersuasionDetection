import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertConfig, BertModel, BertTokenizer


class EncoderImageCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pool_size = cfg['image-model']['grid']

        if 'resnet50' in cfg['image-model']['name']:
            cnn = models.resnet50(pretrained=True)
        elif cfg['image-model']['name'] == 'resnet101':
            cnn = models.resnet101(pretrained=True)

        self.spatial_feats_dim = cnn.fc.in_features
        modules = list(cnn.children())[:-2]
        self.cnn = torch.nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d(pool_size)

    def forward(self, image):
        spatial_features = self.cnn(image)
        spatial_features = self.avgpool(spatial_features)
        return spatial_features


class EncoderTextBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        bert_config = BertConfig.from_pretrained(config['text-model']['pretrain'],
                                                 output_hidden_states=True,
                                                 num_hidden_layers=config['text-model']['extraction-hidden-layer'])
        bert_model = BertModel.from_pretrained(config['text-model']['pretrain'], config=bert_config)

        self.tokenizer = BertTokenizer.from_pretrained(config['text-model']['pretrain'])
        self.bert_model = bert_model

    def forward(self, x, lengths):
        '''
        x: tensor of indexes (LongTensor) obtained with tokenizer.encode() of size B x ?
        lengths: tensor of lengths (LongTensor) of size B
        '''
        max_len = max(lengths)
        attention_mask = torch.ones(x.shape[0], max_len)
        for e, l in zip(attention_mask, lengths):
            e[l:] = 0
        attention_mask = attention_mask.to(x.device)

        outputs = self.bert_model(x, attention_mask=attention_mask)
        outputs = outputs[2][-1]

        return outputs


class PositionalEncodingImageGrid(nn.Module):
    def __init__(self, d_model, n_regions=(4, 4)):
        super().__init__()
        assert n_regions[0] == n_regions[1]
        self.map = nn.Linear(2, d_model)
        self.n_regions = n_regions
        self.coord_tensor = self.build_coord_tensor(n_regions[0])

    @staticmethod
    def build_coord_tensor(d):
        coords = torch.linspace(-1., 1., d)
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)
        ct = torch.stack((x, y), dim=2)
        if torch.cuda.is_available():
            ct = ct.cuda()
        return ct

    def forward(self, x, start_token=False):   # x is seq_len x B x dim
        assert not (start_token and self.n_regions[0] == math.sqrt(x.shape[0]))
        bs = x.shape[1]
        ct = self.coord_tensor.view(self.n_regions[0]**2, -1)   # 16 x 2

        ct = self.map(ct).unsqueeze(1)   # 16 x d_model
        if start_token:
            x[1:] = x[1:] + ct.expand(-1, bs, -1)
            out_grid_point = torch.FloatTensor([-1. - 2/self.n_regions[0], -1.]).unsqueeze(0)
            if torch.cuda.is_available():
                out_grid_point = out_grid_point.cuda()
            x[0:1] = x[0:1] + self.map(out_grid_point)
        else:
            x = x + ct.expand(-1, bs, -1)
        return x


class DualTransformer(nn.Module):
    def __init__(self, cfg, labels):
        super().__init__()
        embed_dim = cfg['model']['embed-dim']
        feedforward_dim = cfg['model']['feedforward-dim']
        num_encoder_layers = cfg['model']['num-encoder-layers']
        num_decoder_layers = cfg['model']['num-decoder-layers']
        visual_features_dim = cfg['image-model']['feat-dim']
        grid = cfg['image-model']['grid']
        self.text_conditioned_on_image_transformer = nn.Transformer(d_model=embed_dim, nhead=8,
                                                                    dim_feedforward=feedforward_dim,
                                                                    dropout=0.1, activation='relu',
                                                                    num_encoder_layers=num_encoder_layers,
                                                                    num_decoder_layers=num_decoder_layers)
        self.image_conditioned_on_text_transformer = nn.Transformer(d_model=embed_dim, nhead=8,
                                                                    dim_feedforward=feedforward_dim,
                                                                    dropout=0.1, activation='relu',
                                                                    num_encoder_layers=num_encoder_layers,
                                                                    num_decoder_layers=num_decoder_layers)

        self.map_text = nn.Linear(cfg['text-model']['word-dim'], embed_dim)
        self.map_image = nn.Linear(visual_features_dim, embed_dim) # + 2 spatial dimensions for encoding the image grid

        self.image_position_conditioner = PositionalEncodingImageGrid(visual_features_dim, grid)

        self.text_multi_label_class_head = nn.Linear(embed_dim, len(labels))
        self.image_multi_label_class_head = nn.Linear(embed_dim, len(labels))


    '''
    boxes: B x S x 4
    embeddings: B x S x dim
    len: B
    targets: ?
    delta_tau: B
    '''
    def forward(self, text, text_len, image):
        bs = text.shape[0]
        device=torch.device("cuda:0")
        #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        text = text.permute(1, 0, 2)    # S x B x dim
        # map text to a common representation space
        text = self.map_text(text)

        image = image.view(bs, image.shape[1], -1).permute(2, 0, 1)  # (d1xd2 x B x dim)

        # augment visual feats with positional info and then map to common representation space
        image = self.image_position_conditioner(image)
        image = self.map_image(image)

        # compute mask for the text (variable length)
        max_text_len = max(text_len)
        txt_mask = torch.ones(bs, max_text_len).bool()
        txt_mask = txt_mask.to(text.device)
        for m, tl in zip(txt_mask, text_len):
            m[:tl] = False

        # forward image transformer conditioned on the text
        image_out = self.image_conditioned_on_text_transformer(src=text, tgt=image, src_key_padding_mask=txt_mask, memory_key_padding_mask=txt_mask)
        hidden_image = image_out[0, :, :]

        # forward text transformer conditioned on the image
        text_out = self.text_conditioned_on_image_transformer(src=image, tgt=text, tgt_key_padding_mask=txt_mask)
        hidden_text = text_out[0, :, :]
        
        contextualized_image_feature = torch.nn.functional.normalize(hidden_image, p=2, dim=-1)
        contextualized_text_feature = torch.nn.functional.normalize(hidden_text, p=2, dim=-1)
 
        return contextualized_image_feature, contextualized_text_feature

class JointTransformerEncoder(nn.Module):
    def __init__(self, cfg, labels):
        super().__init__()
        embed_dim = cfg['model']['embed-dim']
        feedforward_dim = cfg['model']['feedforward-dim']
        num_layers = cfg['model']['num-layers']
        visual_features_dim = cfg['image-model']['feat-dim']
        grid = cfg['image-model']['grid']
        joint_te_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4,
                                                         dim_feedforward=feedforward_dim,
                                                         dropout=0.1, activation='relu')
        self.joint_transformer = nn.TransformerEncoder(joint_te_layer,
                                                       num_layers=num_layers)

        self.map_text = nn.Linear(cfg['text-model']['word-dim'], embed_dim)
        self.map_image = nn.Linear(visual_features_dim, embed_dim) # + 2 spatial dimensions for encoding the image grid

        self.image_position_conditioner = PositionalEncodingImageGrid(visual_features_dim, grid)
#        self.multi_label_class_head = nn.Linear(cfg['model']['embed-dim'], len(labels))
        kernel_num = 32
        kernel_size = [1, 2]
        dropout = 0.1
        self.front = nn.Linear(embed_dim, embed_dim*kernel_num*1)
        self.conv11 = nn.Conv1d(embed_dim , kernel_num, kernel_size[0])
        self.conv12 = nn.Conv1d(embed_dim , kernel_num, kernel_size[1])
        self.dropout = nn.Dropout(dropout)
        self.text_multi_label_class_head = nn.Linear(len(kernel_size) * kernel_num, len(labels))


    def conv_and_pool(self, x, conv):
        #  torch.Size([])
        x = conv(x)
        # 经过一维卷积后的大小 torch.Size([1024,2,1])
        x = F.relu(x)
        # 激活层后：torch.Size([128, 16, 18])
        x = F.max_pool1d(x, x.size(2))  #(128,16,1)  # torch.nn.functional.max_pool1d(input([128, 16, 18]), kernel_size 18)     # x.size(2)指H_out的值
        x = x.squeeze(2)
        #  (batch, kernel_num)   torch.Size([128, 16])         (128,16,1) .squeeze(2)==> (128,16)
        return x


    '''
    boxes: B x S x 4
    embeddings: B x S x dim
    len: B
    targets: ?
    delta_tau: B
    '''
    def forward(self, text, text_len, image):
        bs = text.shape[0]

        text = text.permute(1, 0, 2)    # S x B x dim
        # map text to a common representation space
        text = self.map_text(text)

        if image is not None:
            image = image.view(bs, image.shape[1], -1).permute(2, 0, 1)  # (d1xd2 x B x dim)

            # augment visual feats with positional info and then map to common representation space
            image = self.image_position_conditioner(image)
            image = self.map_image(image)

            # merge image and text features
            image_len = [image.shape[0]] * bs
            embeddings = torch.cat([image, text], dim=0) # S+(d1xd2) x B x dim
        else:
            # only text
            image_len = [0] * bs
            embeddings = text

        # compute mask for the concatenated vector
        max_text_len = max(text_len)
        max_image_len = max(image_len)
        mask = torch.ones(bs, max_text_len + max_image_len).bool()
        mask = mask.to(embeddings.device)
        for m, tl, il in zip(mask, text_len, image_len):
            m[:il] = False
            m[max_image_len:max_image_len + tl] = False

        # forward temporal transformer
        out = self.joint_transformer(embeddings, src_key_padding_mask=mask)
        multimod_feature = out[0, :, :]
        # final multi-class head
        multimod_feature = multimod_feature.unsqueeze(2)
        multimod_feature = multimod_feature.expand(multimod_feature.size()[0], multimod_feature.size()[1], 16)
        x1 = self.conv_and_pool(multimod_feature, self.conv11)
        x2 = self.conv_and_pool(multimod_feature, self.conv12)
        contextualized_text_feature = torch.cat((x1, x2), 1)
        contextualized_text_feature = self.dropout(contextualized_text_feature)
        text_class_logits = self.text_multi_label_class_head(contextualized_text_feature)
        probs = torch.sigmoid(text_class_logits)

#        class_logits = self.multi_label_class_head(multimod_feature)
#        probs = torch.sigmoid(class_logits)
        return probs, multimod_feature


class MemeMultiLabelClassifier(nn.Module):
    def __init__(self, cfg, labels):
        super().__init__()
        self.visual_enabled = cfg['image-model']['enabled'] if 'enabled' in cfg['image-model'] else True
        if self.visual_enabled:
            self.visual_module = EncoderImageCNN(cfg)
        self.textual_module = EncoderTextBERT(cfg)
        if cfg['model']['name'] == 'transformer-encoder' or cfg['model']['name'] == 'transformer':
            self.joint_processing_module = JointTransformerEncoder(cfg, labels)
        elif cfg['model']['name'] == 'dual-transformer':
            self.joint_processing_module = DualTransformer(cfg, labels)

        self.finetune_visual = cfg['image-model']['fine-tune']
        self.finetune_textual = cfg['text-model']['fine-tune']
       # print(len(labels))
       # self.image_feats_fusion = nn.Linear(4096*7*7,2048*7*7)
       # self.inference_threshold = nn.Linear(22,22)
        
#        self.loss = 0 # nn.CrossEntropyLoss()
#        self.labels = labels
        self.criterion = nn.CrossEntropyLoss()#nn.functional.cross_entropy


    def id_to_classes(self, classes_ids):
        out_classes = []
        for elem in classes_ids:
            if elem:
                int_classes = ['1']
            else:
                int_classes = ['0']
#            for idx in enumerate(elem):
#                if ids:
#            int_classes.append(self.labels[elem])
            out_classes.append(int_classes)
        return out_classes

    def _contrastive_loss_forward(self,
                                      hidden_img1: torch.Tensor,
                                      hidden_text1: torch.Tensor,
                                      hidden_img2: torch.Tensor,
                                      hidden_text2: torch.Tensor,
                                      label_org: torch.Tensor,
                                      hidden_norm: bool = True,
                                      temperature: float = 1.0):
        """
        hidden1: (batch_size, dim)
        hidden2: (batch_size, dim)
        """

        LARGE_NUM = 1e9
        batch_size, hidden_dim = hidden_img1.shape
        if hidden_norm:
            hidden_img1 = torch.nn.functional.normalize(hidden_img1, p=2, dim=-1)
            hidden_text1 = torch.nn.functional.normalize(hidden_text1, p=2, dim=-1)
            hidden_img2 = torch.nn.functional.normalize(hidden_img2, p=2, dim=-1)
            hidden_text2 = torch.nn.functional.normalize(hidden_text2, p=2, dim=-1)
        hidden_img1_large = hidden_img1
        hidden_text1_large = hidden_text1
        hidden_img2_large = hidden_img2
        hidden_text2_large = hidden_text2
        labels = torch.arange(0, batch_size).to(device=hidden_img1.device)
        for i in range(batch_size):
            if label_org[i][0] == 0:
                labels[i] = labels[i] + batch_size
#        labels2 = torch.ones(batch_size).to(device=hidden_img1.device)
        masks1 = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(device=hidden_img1.device, dtype=torch.float)
#        masks2 = torch.zeros(batch_size, batch_size).to(device=hidden_img1.device, dtype=torch.float)
        
        logits_aa = torch.matmul(hidden_img1, hidden_text1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_a = torch.abs(logits_aa)
#        logits_aa = #logits_aa - masks1 * LARGE_NUM
#        logits_ab = torch.matmul(hidden_img1, hidden_text1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
#        print(logits_aa)
        logits_bb = torch.matmul(hidden_img2, hidden_text2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_b = torch.abs(logits_bb)
#        logits_bb = logits_bb - masks2 * LARGE_NUM
#        logits_ba = torch.matmul(hidden_img2, hidden_text2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
#        print(logits_bb)
#        labels = torch.cat([labels1, labels2.long()])
#        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
#        logits_b = torch.cat([logits_ba, logits_bb], dim=1)
#        print(logits_a.size())
#        loss = torch.nn.functional.cross_entropy(torch.cat([logits_aa, logits_bb], dim=1), labels)
#        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels2.long())
#        loss = loss_a + loss_b
        logits = torch.cat([logits_aa, logits_bb], dim=1)

#        print(torch.cat([logits_aa, logits_bb], dim=1))
        return logits, labels #loss

    def forward(self, image, text1, text2, text_len1, text_len2, labels=None, return_probs=False, inference_threshold=0.5):
        if self.visual_enabled:
            with torch.set_grad_enabled(self.finetune_visual):
                image_feats = self.visual_module(image)
#                image_feats = torch.cat([image_feats1, image_feats1], 0)

        else:
            image_feats = None
        with torch.set_grad_enabled(self.finetune_textual):
            
            
            text_feats1 = self.textual_module(text1, text_len1)
            text_feats2 = self.textual_module(text2, text_len2)
            
           # for i in range(text_feats.size()[0]):
           #     print(text_feats[i][0][0])
       # print(text_feats.size())
        contextualized_image_feature1, contextualized_text_feature1 = self.joint_processing_module(text_feats1, text_len1, image_feats)
        contextualized_image_feature2, contextualized_text_feature2 = self.joint_processing_module(text_feats2, text_len2, image_feats)

        contrastive_bs = contextualized_image_feature1.size()[0]
        

        if self.training:
#            loss_sum = torch.zeros(contrastive_bs,1)
#            loss = 0
#            loss_sum = loss_sum.cuda()
#            loss = loss.cuda()
#            loss = self.loss(sims[1].unsqueeze(0), labels[1].unsqueeze(0).long())
#            for i in range(sims.size()[1]):
#                for j in range(sims.size()[0]):
#                    loss_sum[i] = loss_sum[i] + sims[j][i]
#                loss = loss + (sims[i][i]/loss_sum[i])
            sim, label = self._contrastive_loss_forward(contextualized_image_feature1, contextualized_text_feature1, contextualized_image_feature2, contextualized_text_feature2, labels, hidden_norm=False, temperature=0.5)
            loss = self.criterion(sim, label)
#            print(loss)
            return loss/10
        else:
            # probs = F.sigmoid(class_logits)
            if return_probs:
                return contextualized_image_feature1, contextualized_text_feature1, contextualized_image_feature2, contextualized_text_feature2
#            print(probs)
            sim1 = torch.ones(contextualized_image_feature1.size()[0])
            sim2 = torch.ones(contextualized_image_feature1.size()[0])
            for j in range(contextualized_image_feature1.size()[0]):
                sim1[j] = torch.dot(contextualized_image_feature1[j], contextualized_text_feature1[j])
                sim2[j] = torch.dot(contextualized_image_feature2[j], contextualized_text_feature2[j])
            classes_ids = sim1 > sim2

            classes = self.id_to_classes(classes_ids)
            return classes
