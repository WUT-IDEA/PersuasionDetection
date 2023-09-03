import argparse
import os
import json
import numpy as np

import torch
import tqdm
from torch.utils.data import DataLoader

import models
from dataset import SemEvalDataset, Collate
from format_checker.task1_3 import read_classes, check_format_task1_task3
from scorer.task1_3 import evaluate

from torchvision import transforms as T


def id_to_classes(classes_ids, labels):
    out_classes = []
    for elem in classes_ids:
#        int_classes = []
#        for idx, ids in enumerate(elem):
        if elem:
            int_classes = ["1"]
        else:
            int_classes = ["0"]
        out_classes.append(int_classes)
    return out_classes

def main(opt):
    checkpoint = torch.load('runs/test/model_best_fold0.pt', map_location='cpu')
    cfg = checkpoint['cfg']
    if 'task' not in cfg['dataset']:
        cfg['dataset']['task'] = 3 # for back compatibility
        print('Manually assigning: task 3')

    if cfg['dataset']['task'] == 3:
        classes = read_classes('techniques_list_task3.txt')
    elif cfg['dataset']['task'] == 1:
        classes = read_classes('techniques_list_task1-2.txt')

    if opt.ensemble or opt.cross_validation:
        checkpoints_folder = os.path.split(opt.checkpoint)[0]
        checkpoints_files = [os.path.join(checkpoints_folder, f) for f in os.listdir(checkpoints_folder) if '.pt' in f]
    else:
        checkpoints_files = [opt.checkpoint]

    ensemble_models = []
    for chkp in checkpoints_files:
        model = models.MemeMultiLabelClassifier(cfg, classes)
        checkpoint = torch.load('runs/test/model_best_fold0.pt', map_location='cpu')
        # Load weights to resume from
        if not cfg['text-model']['fine-tune'] and not cfg['image-model']['fine-tune']:
            # the visual and textual modules are already fine
            model.joint_processing_module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        model.cuda().eval()
        ensemble_models.append(model)

    # Load data loaders
    test_transforms = T.Compose([T.Resize(256),
                                 T.CenterCrop(224),
                                 T.ToTensor(),
                                 T.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
    collate_fn = Collate(cfg, classes)

    if opt.cross_validation:
        datasets = [SemEvalDataset(cfg, split='val', transforms=test_transforms, val_fold=fold) for fold in range(len(checkpoints_files))]
        dataloaders = [DataLoader(dataset, batch_size=8, shuffle=False,
                                num_workers=2, collate_fn=collate_fn) for dataset in datasets]
    else:
        if opt.validate:
            dataset = SemEvalDataset(cfg, split='val', transforms=test_transforms, val_fold=opt.val_fold)
        elif opt.test:
            dataset = SemEvalDataset(cfg, split='test', transforms=test_transforms, val_fold=opt.val_fold)
        else:
            dataset = SemEvalDataset(cfg, split='dev', transforms=test_transforms)

        dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
                                      num_workers=2, collate_fn=collate_fn)
        dataloaders = [dataloader]

    resumed_logdir, resumed_filename = os.path.split('runs/test/model_best_fold0.pt')
    del checkpoint  # current, saved
    print('Model {} resumed from {}, saving results on this directory...'.format(resumed_filename, resumed_logdir))

    predictions = {} if opt.cross_validation else []
    metrics = {}
    thr = opt.threshold
    if opt.cross_validation:
        for idx, (dataloader, model) in enumerate(tqdm.tqdm(zip(dataloaders, ensemble_models))):
            for it, (image, text1, text2, text_len1, text_len2, labels, ids) in enumerate(tqdm.tqdm(dataloader)):
                if torch.cuda.is_available():
                    image = image.cuda() if image is not None else None
                    text1 = text1.cuda()
                    text2 = text2.cuda()
                   # labels = labels.cuda()

                # cross-validation
                with torch.no_grad():
                    contextualized_image_feature1, contextualized_text_feature1, contextualized_image_feature2, contextualized_text_feature2 = model(image, text1, text2, text_len1, text_len2, return_probs=True)

                    image_feature1 = torch.nn.functional.normalize(contextualized_image_feature1, p=2, dim=-1)
                    text_feature1 = torch.nn.functional.normalize(contextualized_text_feature1, p=2, dim=-1)
                    image_feature2 = torch.nn.functional.normalize(contextualized_image_feature2, p=2, dim=-1)
                    text_feature2 = torch.nn.functional.normalize(contextualized_text_feature2, p=2, dim=-1)

                    for j in range(contextualized_image_feature1.size()[0]):
                        sim1[j] = dot(image_feature1[j], text_feature1[j])
                        sim2[j] = dot(image_feature2[j], text_feature2[j])
                        valid_pred[j][0] = sim1[j] > sim2[j]
                  #  for i in range(probs.size()[0]):
                  #       print(pred_probs[i][16])
                  #       valid_pred[i][16] = pred_probs[i][16] > thr+0.1
                    pred_classes = id_to_classes(valid_pred, classes)

                    for id, labels in zip(ids, pred_classes):  # loop over every element of the batch
                        if idx not in predictions:
                            predictions[idx] = []
                        predictions[idx].append({'id': id, 'labels': labels})
    else:
        for it, (image, text1, text2, text_len1, text_len2, labels, ids) in enumerate(tqdm.tqdm(dataloader)):
            if torch.cuda.is_available():
                image = image.cuda() if image is not None else None
                text1 = text1.cuda()
                text2 = text2.cuda()
                # labels = labels.cuda()

            ensemble_predictions = []
            with torch.no_grad():
                for model in ensemble_models:
                    contextualized_image_feature1, contextualized_text_feature1, contextualized_image_feature2, contextualized_text_feature2 = model(image, text1, text2, text_len1, text_len2, return_probs=True)

                    image_feature1 = torch.nn.functional.normalize(contextualized_image_feature1, p=2, dim=-1)
                    text_feature1 = torch.nn.functional.normalize(contextualized_text_feature1, p=2, dim=-1)
                    image_feature2 = torch.nn.functional.normalize(contextualized_image_feature2, p=2, dim=-1)
                    text_feature2 = torch.nn.functional.normalize(contextualized_text_feature2, p=2, dim=-1)
                    sim1 = torch.ones(contextualized_image_feature1.size()[0])
                    sim2 = torch.ones(contextualized_image_feature1.size()[0])
#                    class_ensemble = torch.bool(contextualized_image_feature1.size()[0],2)
                    for j in range(contextualized_image_feature1.size()[0]):
                        sim1[j] = torch.dot(image_feature1[j], text_feature1[j])
                        sim2[j] = torch.dot(image_feature2[j], text_feature2[j])
                    class_ensemble = sim1 > sim2
#                    class_ensemble2 = sim1 < sim2
#                    class_ensemble = torch.stack((class_ensemble1, class_ensemble2), dim=1)
#                    print(sim1)
#                    print(222222)
#                    print(sim2)


#                for i in range(prob_ensemble.size()[0]):
                   # print(class_ensemble[i][16])
#                    class_ensemble[i][0] = prob_ensemble[i][0] > 0
#                    class_ensemble[i][2] = prob_ensemble[i][2] > thr - 0.11
#                    class_ensemble[i][5] = prob_ensemble[i][5] > thr - 0.1
#                    class_ensemble[i][6] = prob_ensemble[i][6] > thr - 0.22
#                    class_ensemble[i][7] = prob_ensemble[i][7] > thr - 0.11
#                    class_ensemble[i][8] = prob_ensemble[i][8] > thr - 0.21
#                    class_ensemble[i][9] = prob_ensemble[i][9] > thr + 0.25
#                    class_ensemble[i][10] = prob_ensemble[i][10] > thr + 0.23
#                    class_ensemble[i][13] = prob_ensemble[i][13] > thr - 0.22
#                    class_ensemble[i][15] = prob_ensemble[i][15] > thr - 0.19
#                    class_ensemble[i][16] = prob_ensemble[i][16] > thr - 0.25
#                    class_ensemble[i][18] = prob_ensemble[i][18] > thr + 0.05
#                    class_ensemble[i][20] = prob_ensemble[i][20] >thr + 0.35

#                    class_ensemble[i][17] = prob_ensemble[i][17] >0.08
#                    class_ensemble[i][18] = prob_ensemble[i][18] >0.5
#                    class_ensemble[i][20] = prob_ensemble[i][20] > 0.08
#                    class_ensemble[i][3] = prob_ensemble[i][3] > 0.27


                    print(class_ensemble)
                pred_classes = id_to_classes(class_ensemble, classes)

            for id, labels in zip(ids, pred_classes):    # loop over every element of the batch
                predictions.append({'id': id, 'labels': labels})

    if opt.cross_validation:
        mean_macro_f1 = 0
        mean_micro_f1 = 0
        for k in range(len(predictions)):
            dataloader = dataloaders[k]
            preds = predictions[k]
            macro_f1, micro_f1 = evaluate(preds, dataloader.dataset.targets, classes)
            mean_macro_f1 += macro_f1
            mean_micro_f1 += micro_f1
        mean_micro_f1 /= len(predictions)
        mean_macro_f1 /= len(predictions)
        out_string = 'Mean-MacroF1: {}\nMean-MicroF1: {}'.format(mean_macro_f1, mean_micro_f1)
        out_file = os.path.join(resumed_logdir, 'cross_validation_results.log')
        with open(out_file, 'w') as f:
            f.write(out_string)
        print(out_string)

    elif opt.validate:
        macro_f1, micro_f1 = evaluate(predictions, dataloader.dataset.targets, classes)
        print('MacroF1: {}\nMicroF1: {}'.format(macro_f1, micro_f1))
    else:
        # dump predictions on json file
        out_json = os.path.join(resumed_logdir, 'predictions_thr{}.json'.format(thr))
        with open(out_json, 'w') as f:
            json.dump(predictions, f)

        # cross check
        if not check_format_task1_task3(out_json, CLASSES=classes):
            print('Saved file has incorrect format! Retry...')
        print('Detection dumped on {}'.format(out_json))
        print('Num memes: {}'.format(len(predictions)))

    print('DONE!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', default=0.3, type=float, help="Threshold to use for classification")
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads only the model')
    parser.add_argument('--validate', action='store_true', help="If not set, default is inference on the dev set")
    parser.add_argument('--test', action='store_true', help="If set, performs inference on the test set")
    parser.add_argument('--val_fold', default=0, type=int, help="Which fold we validate on (use with --validate)")
    parser.add_argument('--ensemble', action='store_true', help='Enables model ensembling')
    parser.add_argument('--cross-validation', action='store_true', help='Enables model ensembling')
    # parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")

    opt = parser.parse_args()
    opt.validate = opt.validate | opt.cross_validation
    print(opt)

    main(opt)
