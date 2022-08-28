import re
import argparse
import os
import shutil
import time
import math
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from helpers import *
from CALP import *
args = None
best_prec1 = 0
global_step = 0

def run_at_once(IsCALP):
    global global_step
    global best_prec1
    # 模型名字
    if args.isMT:
        model_name = '%s_%d_mt_ss_split_%d_isL2_%d_IsCALP_%d' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2),int(IsCALP))
    else:
        model_name = '%s_%d_ss_split_%d_isL2_%d_IsCALP_%d' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2),int(IsCALP))
    #保存训练模型的路径
    checkpoint_path = 'verify/%s' % model_name
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    #确定保存路径的文件
    log_file = '%s/log.txt' % checkpoint_path
    log = open(log_file, 'a')

    # 配置dataset的路径
    dataset_config = datasets.__dict__[args.dataset](isTwice=args.isMT)
    num_classes = dataset_config.pop('num_classes')

    #配置文件加载
    train_loader, eval_loader, train_loader_noshuff, train_data = create_data_loaders(**dataset_config, args=args)

    #创建模型
    model = create_model(num_classes,args)

    #判断是否夹加入MT
    if args.isMT:
        ema_model = create_model(num_classes,args,ema=True)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    cudnn.benchmark = True

    # 载入模型
    if args.isMT:
    	resume_fn = 'models/%s_%d_mean_teacher_split_%d_isL2_%d/checkpoint.180.ckpt' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2))
    else:
    	resume_fn = 'models/%s_%d_split_%d_isL2_%d/checkpoint.180.ckpt' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2))

    # 载入模型
    assert os.path.isfile(resume_fn), "=> no checkpoint found at '{}'".format(resume_fn)
    checkpoint = torch.load(resume_fn)
    best_prec1 = checkpoint['best_prec1']

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

    # 计算初始的正确率
    prec1, prec5 = validate(eval_loader, model, global_step, args.start_epoch, isMT = args.isMT)
    if args.isMT:
        ema_prec1, ema_prec5  = validate(eval_loader, ema_model, global_step, args.start_epoch, isMT = args.isMT)

    print('Resuming from:%s' % resume_fn)

    #开始训练
    IsCALP=True
    for epoch in range(args.start_epoch, args.epochs):
        if epoch==args.start_epoch:
            # Extract features and update the pseudolabels
            print('Extracting features...') #提取特征的时候用train_loader_noshuff速度更快
            feats = extract_features(train_loader_noshuff, model, isMT = args.isMT)
            #train_data更新了
            sel_acc,weight,p_labels = update_plabels(train_data,feats, k=args.dfs_k, max_iter = 20)

        if epoch>args.start_epoch:
            # 提取特征
            print('Extracting features...')
            feats = extract_features(train_loader_noshuff, model, isMT=args.isMT)
            if IsCALP==True:
               Expand_batch_features,anchor_y,anchor_n=Expand_Positive_Negtive_anchor(feats, p_labels, weight, num_classes)
               sel_acc, weight, p_labels  = update_plabels(train_data,Expand_batch_features, k=args.dfs_k, max_iter=20, IsCALP=True)
            else:
               sel_acc, weight, p_labels =  update_plabels(train_data,feats, k=args.dfs_k, max_iter=20)

        #  更新网络
        if args.isMT:
            train_meter, global_step = train(train_loader, model, optimizer, epoch, global_step, args,
                                                 ema_model=ema_model)
        else:
            train_meter, global_step = train(train_loader, model, optimizer, epoch, global_step, args)


         # 测试数据的验证
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            print("Evaluating the primary model:")
            prec1, prec5 = validate(eval_loader, model, global_step, epoch + 1, isMT = args.isMT)

            if args.isMT:
                print("Evaluating the EMA model:")
                ema_prec1, ema_prec5  = validate(eval_loader, ema_model, global_step, epoch + 1, isMT = args.isMT)
                is_best = ema_prec1 > best_prec1
                best_prec1 = max(ema_prec1, best_prec1)
            else:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
        else:
            is_best = False

        # 记录
        if args.isMT:
            log.write('%d\t%.4f\t%.4f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' %
                (epoch,
                train_meter['class_loss'].avg,
                train_meter['lr'].avg,
                train_meter['top1'].avg,
                train_meter['top5'].avg,
                prec1,
                prec5,
                ema_prec1,
                ema_prec5)
            )
            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_path, epoch + 1)

        else:
            log.write('%d,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f\n' %
                (epoch,
                train_meter['class_loss'].avg,
                train_meter['lr'].avg,
                train_meter['top1'].avg, #表示为训练集的Top1准确率
                train_meter['top5'].avg, #表示为训练集的Top5准确率
                prec1,                   #表示为测试集的Top1准确率
                prec5,                   #表示为测试集的Top5准确率
                sel_acc                  #表示为伪标签的Top1准确率
                )
            )
            Incorrect_max_w, Incorrect_max_w_imgpath, correct_max_w,correct_max_w_imgpath= show_maxweightimage(train_data.p_weights, train_data.all_labels, train_data.p_labels,
                                                       train_data.imgs)
            rawdatawritelogger(Incorrect_max_w, '%s/Incorrect_max_w.txt' % checkpoint_path)
            rawdatawritelogger(Incorrect_max_w_imgpath, '%s/Incorrect_max_w_imgpath.txt' % checkpoint_path)
            rawdatawritelogger(correct_max_w, '%s/correct_max_w.txt' % checkpoint_path)
            rawdatawritelogger(correct_max_w_imgpath, '%s/correct_max_w_imgpath.txt' % checkpoint_path)

            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_path, epoch + 1)





if __name__ == '__main__':
    # Get the command line arguments
    args = cli.parse_commandline_args()
    #for times in range(args.label_split):
      #args.label_split=times+10
      # Set the other settings
      #args = load_args(args, isMT = args.isMT)
      # Use only the specified GPU
      #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
      #print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled,args.label_split,args.gpu_id))
      #run_at_once(False)
      #run_at_once(True)
    args = load_args(args, isMT = args.isMT)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled, args.label_split, args.gpu_id))
    run_at_once(True)
    run_at_once(False)