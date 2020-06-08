import torch
import torch.optim as optim
from tqdm import tqdm
import time
import random
import os
import sys

from config import *
from volleyball import *
from dataset import *
from gcn_model import *
from base_model import *
from utils import *

import pdb


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train_net(cfg):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)

    # Reading dataset
    training_set, validation_set1, validation_set2, validation_set3, validation_set = return_dataset(cfg)
    # training_set, validation_set = return_dataset(cfg)

    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 8
    }
    training_loader = data.DataLoader(training_set, **params)

    params['batch_size'] = cfg.test_batch_size
    validation_loader = data.DataLoader(validation_set, **params)
    validation_loader1 = data.DataLoader(validation_set1, **params)
    validation_loader2 = data.DataLoader(validation_set2, **params)
    validation_loader3 = data.DataLoader(validation_set3, **params)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Build model and optimizer
    basenet_list = {'volleyball': Basenet_volleyball}
    gcnnet_list = {'volleyball': GCNnet_volleyball}

    if cfg.training_stage == 1:
        Basenet = basenet_list[cfg.dataset_name]
        model = Basenet(cfg)
        # model.loadmodel(cfg.stage1_model_path)
    elif cfg.training_stage == 2:
        GCNnet = gcnnet_list[cfg.dataset_name]
        model = GCNnet(cfg)
        model.cuda()
        # Load backbone
        model.loadmodel(cfg.stage1_model_path)
        # state_dict=torch.load('/home/hanbin/CurrentProject/Group-Activity-Recognition/scripts/result/[Volleyball_stage2_stage2]<2020-03-16_05-50-30>/stage2_epoch14_53.76.pth')
        # model.load_state_dict(state_dict['state_dict'])

    else:
        assert (False)


    if cfg.use_multi_gpu:
        model = nn.DataParallel(model)

    model = model.to(device=device)

    model.train()
    model.apply(set_bn_eval)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train_learning_rate,
                           weight_decay=cfg.weight_decay)

    train_list = {'volleyball': train_volleyball}
    test_list = {'volleyball': test_volleyball}
    train = train_list[cfg.dataset_name]
    test = test_list[cfg.dataset_name]

    if cfg.test_before_train:

        state_dict=torch.load('result/[Volleyball_stage2_stage2]<2020-03-21_22-01-20>/stage2_epoch5_77.32%.pth')
        model.load_state_dict(state_dict['state_dict'])

        test_info = test(validation_loader1, model, device, 0, cfg)
        print("OR=0.1")
        print(test_info)
        test_info = test(validation_loader2, model, device, 0, cfg)
        print("OR=0.4")
        print(test_info)
        test_info = test(validation_loader3, model, device, 0, cfg)
        print("OR=0.7")
        print(test_info)
        test_info = test(validation_loader, model, device, 0, cfg)
        print("OR=1")
        print(test_info)        
        return

    # Training iteration
    best_result = {'epoch': 0, 'activities_acc': 0}
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):

        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])

        # One epoch of forward and backward
        # test_info = test(validation_loader, model, device, epoch, cfg)
        train_info = train(training_loader, model, device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info = test(validation_loader, model, device, epoch, cfg)
            show_epoch_info('Test', cfg.log_path, test_info)

            if test_info['activities_acc'] > best_result['activities_acc']:
                best_result = test_info
            print_log(cfg.log_path,
                      'Best group activity accuracy: %.2f%% at epoch #%d.' % (
                      best_result['activities_acc'], best_result['epoch']))

            # Save model
            if cfg.training_stage == 2:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                filepath = cfg.result_path + '/stage%d_epoch%d_%.2f%%.pth' % (
                cfg.training_stage, epoch, test_info['activities_acc'])
                torch.save(state, filepath)
                print('model saved to:', filepath)
            elif cfg.training_stage == 1:
                for m in model.modules():
                    if isinstance(m, Basenet):
                        filepath = cfg.result_path + '/stage%d_epoch%d_%.2f%%.pth' % (
                        cfg.training_stage, epoch, test_info['activities_acc'])
                        m.savemodel(filepath)
            #                         print('model saved to:',filepath)
            else:
                assert False


def train_volleyball(data_loader, model, device, optimizer, epoch, cfg):
    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    for batch_data in tqdm(data_loader):
        model.train()
        model.apply(set_bn_eval)

        # prepare batch data
        batch_data = [b.to(device=device) for b in batch_data]
        # print("batch_data_test", batch_data[0].shape)
        # print("batch_data", batch_data[1].shape)

        batch_size = batch_data[0].shape[0]
        num_frames = batch_data[0].shape[1]

        # actions_in = batch_data[2].reshape((batch_size, num_frames, cfg.num_boxes))
        activities_in = batch_data[2].reshape((batch_size, num_frames))

        # actions_in = actions_in[:, 0, :].reshape((batch_size * cfg.num_boxes,))
        activities_in = activities_in[:, 0].reshape((batch_size,))

        # forward
        actions_scores, activities_scores = model((batch_data[0], batch_data[1]))
        # print(activities_scores)

        # Predict actions
        actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
        # actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
        actions_labels = torch.argmax(actions_scores, dim=1)
        # actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())

        # Predict activities
        # print(activities_scores)
        activities_loss = F.cross_entropy(activities_scores, activities_in)
        activities_labels = torch.argmax(activities_scores, dim=1)
        activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())

        # Get accuracy
        # actions_accuracy = actions_correct.item() / actions_scores.shape[0]
        activities_accuracy = activities_correct.item() / activities_scores.shape[0]

        # actions_meter.update(actions_accuracy, actions_scores.shape[0])
        activities_meter.update(activities_accuracy, activities_scores.shape[0])

        # Total loss
        total_loss = activities_loss  # +cfg.actions_loss_weight*actions_loss
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'activities_acc': activities_meter.avg * 100,
        # 'actions_acc': actions_meter.avg * 100
    }

    return train_info


def test_volleyball(data_loader, model, device, epoch, cfg):
    model.eval()

    actions_meter = AverageMeter()
    activities_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()
    activities_subset_correct = {}
    activities_subset_total = {}
    activities_subset_ptotal={}
    for i in range(18):
        activities_subset_correct[str(i)] = 0
        activities_subset_total[str(i)] = 0
        activities_subset_ptotal[str(i)] = 0
    with torch.no_grad():
        for batch_data_test in tqdm(data_loader):
            # prepare batch data
            batch_data_test = [b.to(device=device) for b in batch_data_test]

            batch_size = batch_data_test[0].shape[0]
            num_frames = batch_data_test[0].shape[1]

            # actions_in = batch_data_test[2].reshape((batch_size, num_frames, cfg.num_boxes))
            activities_in = batch_data_test[2].reshape((batch_size, num_frames))

            # forward
            actions_scores, activities_scores = model((batch_data_test[0], batch_data_test[1]))

            # Predict actions
            # actions_in = actions_in[:, 0, :].reshape((batch_size * cfg.num_boxes,))
            activities_in = activities_in[:, 0].reshape((batch_size,))

            actions_weights = torch.tensor(cfg.actions_weights).to(device=device)
            # actions_loss = F.cross_entropy(actions_scores, actions_in, weight=actions_weights)
            actions_labels = torch.argmax(actions_scores, dim=1)

            # Predict activities
            activities_loss = F.cross_entropy(activities_scores, activities_in)
            activities_labels = torch.argmax(activities_scores, dim=1)

            # actions_correct = torch.sum(torch.eq(actions_labels.int(), actions_in.int()).float())
            activities_correct = torch.sum(torch.eq(activities_labels.int(), activities_in.int()).float())

            pred = activities_labels.data.cpu().numpy()
            labels = activities_in.data.cpu().numpy()
            # print(labels,pred)
            for i in range(len(labels)):
                activities_subset_total[str(labels[i])] += 1
                activities_subset_ptotal[str(pred[i])] += 1
                if pred[i] == labels[i]:
                    activities_subset_correct[str(labels[i])] += 1

            # activities_subset_labels=torch.chunk(activities_labels,chunks=activities_labels.size(1),dim=1)
            # activities_subset_in = torch.chunk(activities_in, chunks=activities_in.size(1), dim=1)
            # for i in range(18):
            #     activities_subset_correct=torch.eq(activities_subset_labels[i].int(),activities_subset_in[i].int()).float()
            #     activities_subset_accuracy=activities_subset_correct.item()/activities_scores.shape[0]
            #     activities_subset_meter[str(i)].update(activities_subset_accuracy,activities_scores.shape[0])
            # activities_subset_accuracy = activities_subset_correct.item()
            # print(activities_subset_correct)

            # Get accuracy
            # actions_accuracy = actions_correct.item() / actions_scores.shape[0]
            activities_accuracy = activities_correct.item() / activities_scores.shape[0]

            # actions_meter.update(actions_accuracy, actions_scores.shape[0])
            activities_meter.update(activities_accuracy, activities_scores.shape[0])

            # Total loss
            total_loss = activities_loss #+ cfg.actions_loss_weight * actions_loss
            loss_meter.update(total_loss.item(), batch_size)
    # ACTIVITIES = ['r-serve', 'r-block', 'r-firstpass', 'r-set', 'r-spike', 'r-winpoint', 'r-drop', 'r-volley', 'r-shot',
    #               'l-serve', 'l-block', 'l-firstpass', 'l-set', 'l-spike', 'l-winpoint', 'l-drop', 'l-volley', 'l-shot']
    # for i in range(18):
    #     print(ACTIVITIES[i] + ':',
    #           np.float(activities_subset_correct[str(i)]) / activities_subset_total[str(i)],
    #           activities_subset_correct[str(i)],
    #           activities_subset_total[str(i)],
    #           activities_subset_ptotal[str(i)])
    # print(activities_subset_correct,activities_subset_total)
    test_info = {
        #'time': epoch_timer.timeit(),
        #'epoch': epoch,
        #'loss': loss_meter.avg,
        'winner_acc': activities_meter.avg * 100,
        # 'actions_acc': actions_meter.avg * 100
    }

    return test_info


