import os

import tqdm
import yaml
import json
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

import torch

import logging
from torch.utils.tensorboard import SummaryWriter

import argparse

from dataset import SemEvalDataset, Collate
from models import MemeMultiLabelClassifier
from sampler import MultilabelBalancedRandomSampler

from task1_3 import evaluate
from format_checker.task1_3 import read_classes
from shutil import copyfile


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='Number of training epochs.')
    # parser.add_argument('--crop_size', default=224, type=int,
    #                     help='Size of an image crop as the CNN input.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=200, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--test_step', default=100000000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/test',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads model, optimizer, scheduler')
    parser.add_argument('--load-model', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads only the model')
    parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")
    parser.add_argument('--cross-validation', action='store_true', help='Enables cross validation')

    opt = parser.parse_args()
    print(opt)
    with open('cfg/config_dual_transformer_task3.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile)

    if opt.cross_validation:
        # read splits from file
        with open('data/folds.json', 'r') as f:
            folds = json.load(f)
            num_folds = len(folds)
        for fold in tqdm.trange(num_folds):
            train(opt, config, val_fold=fold)
    else:
        # train using fold 0 as validation fold
        train(opt, config, val_fold=0)

def train(opt, config, val_fold=0):
    # torch.cuda.set_enabled_lms(True)
    # if (torch.cuda.get_enabled_lms()):
    #     torch.cuda.set_limit_lms(11000 * 1024 * 1024)
    #     print('[LMS=On limit=' + str(torch.cuda.get_limit_lms()) + ']')

    if 'task' not in config['dataset']:
        config['dataset']['task'] = 3 # for back compatibility
        print('Manually assigning: task 3')

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger = SummaryWriter(log_dir=opt.logger_name, comment='')
    experiment_path = tb_logger.get_logdir()

    # Dump configuration to experiment path
    copyfile('cfg/config_dual_transformer_task3.yaml', os.path.join(experiment_path, 'config.json'))

    # Load Vocabulary Wrapper

    # Load data loaders
    test_transforms = T.Compose([T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    train_transforms = T.Compose([T.Resize(256),
                    T.RandomCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    train_dataset = SemEvalDataset(config, split='train', transforms=train_transforms, val_fold=val_fold)
    val_dataset = SemEvalDataset(config, split='dev', transforms=test_transforms, val_fold=val_fold)

    id_intersection = set([x['id'] for x in train_dataset.targets]).intersection([x['id'] for x in val_dataset.targets])

#    assert len(id_intersection) == 0

    if config['dataset']['task'] == 3:
        classes = read_classes('techniques_list_task3.txt')
    elif config['dataset']['task'] == 1:
        classes = read_classes('techniques_list_task1-2.txt')
    print(config)
    collate_fn = Collate(config, classes)
    if 'balanced-sampling' in config['training'] and config['training']['balanced-sampling']:
        classes_ids = [[train_dataset.class_list.index(x) for x in info['labels']] for info in train_dataset.targets]
        labels = np.zeros((len(classes_ids), len(train_dataset.class_list)))
        for l, c in zip(labels, classes_ids):
            l[c] = 1
        sampler = MultilabelBalancedRandomSampler(labels)
    else:
        sampler = None

    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True if sampler is None else False, num_workers=opt.workers, collate_fn=collate_fn, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['bs'], shuffle=False,
                                  num_workers=opt.workers, collate_fn=collate_fn)

    # Construct the model
    model = MemeMultiLabelClassifier(config, labels=classes)
    if torch.cuda.is_available() and not (opt.resume or opt.load_model):
        model.cuda()

    # Construct the optimizer
    if not config['text-model']['fine-tune'] and not config['image-model']['fine-tune']:
        optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if 'textual_module' not in n and 'visual_module' not in n], lr=config['training']['lr'])
    else:
        if config['dataset']['task'] == 3:
            optimizer = torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'textual_module' not in n and 'visual_module' not in n]},
                {'params': model.textual_module.parameters(), 'lr': config['training']['pretrained-modules-lr']},
                {'params': model.visual_module.parameters(), 'lr': config['training']['pretrained-modules-lr']}]
                , lr=config['training']['lr'])
        elif config['dataset']['task'] == 1:
            optimizer = torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if
                            'textual_module' not in n and 'visual_module' not in n]},
                {'params': model.textual_module.parameters(), 'lr': config['training']['pretrained-modules-lr']}]
                , lr=config['training']['lr'])
    # LR scheduler
    scheduler_name = config['training']['scheduler']
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=config['training']['gamma'], milestones=config['training']['milestones'])
    elif scheduler_name is None:
        scheduler = None
    else:
        raise ValueError('{} scheduler is not available'.format(scheduler_name))


    # # optionally resume from a checkpoint
    start_epoch = 0
    # if opt.resume or opt.load_model:
    #     filename = opt.resume if opt.resume else opt.load_model
    #     if os.path.isfile(filename):
    #         print("=> loading checkpoint '{}'".format(filename))
    #         checkpoint = torch.load(filename, map_location='cpu')
    #         model.load_state_dict(checkpoint['model'], strict=False)
    #         if torch.cuda.is_available():
    #             model.cuda()
    #         if opt.resume:
    #             start_epoch = checkpoint['epoch']
    #             # best_rsum = checkpoint['best_rsum']
    #             optimizer.load_state_dict(checkpoint['optimizer'])
    #             if checkpoint['scheduler'] is not None and not opt.reinitialize_scheduler:
    #                 scheduler.load_state_dict(checkpoint['scheduler'])
    #             # Eiters is used to show logs as the continuation of another
    #             # training
    #             model.Eiters = checkpoint['Eiters']
    #             print("=> loaded checkpoint '{}' (epoch {})"
    #                   .format(opt.resume, start_epoch))
    #         else:
    #             print("=> loaded only model from checkpoint '{}'"
    #                   .format(opt.load_model))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(opt.resume))

    model.train()

    # Train loop
    mean_loss = 0
    progress_bar = tqdm.trange(start_epoch, opt.num_epochs)
    progress_bar.set_description('Train')
    best_f11 = 0.0
    for epoch in progress_bar:
        for it, (image, text1, text2, text_len1, text_len2, labels, ids) in enumerate(train_dataloader):
            global_iteration = epoch * len(train_dataloader) + it
            label = torch.eye(2*labels.size()[0], labels.size()[0])
            if torch.cuda.is_available():
                image = image.cuda() if image is not None else None
                text1 = text1.cuda()
                text2 = text2.cuda()
                labels = labels.cuda()

            # forward the model
            optimizer.zero_grad()

            loss = model(image, text1, text2, text_len1, text_len2, labels)
#            print('loss:',loss)
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()

            if global_iteration % opt.log_step == 0:
                mean_loss /= opt.log_step
                progress_bar.set_postfix(dict(loss='{:.2}'.format(mean_loss)))
                mean_loss = 0

            tb_logger.add_scalar("Training/Epoch", epoch, global_iteration)
            tb_logger.add_scalar("Training/Loss", loss.item(), global_iteration)
            tb_logger.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], global_iteration)

            if global_iteration % opt.val_step == 0:
                # validate (using different thresholds)

              #  inthre1=torch.Tensor(np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]))

                metrics = validate(val_dataloader, model, classes, thresholds=[0.25]) #, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
#                tb_logger.add_scalars("Validation/F1", metrics, global_iteration)
#                print(metrics)
                #progress_bar.set_postfix(dict(macroF1='{:.2}'.format(metrics['macroF1_thr=0.5']), microF1='{:.2}'.format(metrics['microF1_thr=0.5'])))
                
                # save best model
                print(metrics)
                if metrics['macroF1_thr=0.25'] > best_f11: #metrics['microF1_thr=0.3'] + 
                    print('Saving 1 best model...')
                    checkpoint = {
                            'cfg': config,
                            'epoch': epoch,
                            'model': model.joint_processing_module.state_dict() if not config['text-model']['fine-tune'] and not config['image-model']['fine-tune'] else model.state_dict()}
                        # 'optimizer': optimizer.state_dict(),
                        # 'scheduler': scheduler.state_dict()}
                    latest = os.path.join(experiment_path, 'model_best_fold{}.pt'.format(val_fold))
                    torch.save(checkpoint, latest)
                    best_f11 = metrics['macroF1_thr=0.25'] #0.2
                    print(111111,best_f11)


        scheduler.step()


def validate(val_dataloader, model, classes_list, thresholds=[0.3]):
    model.eval()
    predictions = []
    metrics = {}
    progress_bar = tqdm.tqdm(thresholds)
    progress_bar.set_description('Validation')
    for thr in progress_bar:
        for it, (image, text1, text2, text_len1, text_len2, labels, ids) in enumerate(val_dataloader):
            label = torch.eye(2*labels.size()[0], labels.size()[0])
            if torch.cuda.is_available():
                image = image.cuda() if image is not None else None
                text1 = text1.cuda()
                text2 = text2.cuda()
                labels = labels.cuda()
               # inthre1 = inthre1.cuda()
            with torch.no_grad():
                pred_classes = model(image, text1, text2, text_len1, text_len2, inference_threshold=thr)

            for id, labels in zip(ids, pred_classes):    # loop over every element of the batch
                predictions.append({'id': id, 'labels': labels})
#        print(predictions)
#        print(val_dataloader.dataset.targets) 
        macro_f1 = evaluate(predictions, val_dataloader.dataset.targets, classes_list)
        metrics['macroF1_thr={}'.format(thr)] = macro_f1
#        metrics['microF1_thr={}'.format(thr)] = micro_f1
#        metrics = micro_f1[1]
#        print(metrics)
    model.train()
    return metrics

if __name__ == '__main__':
    main()

