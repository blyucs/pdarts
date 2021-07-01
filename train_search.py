import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import copy
from model_search import Network
from model import NetworkCIFAR
from genotypes import PRIMITIVES_NORMAL,PRIMITIVES_REDUCE, NORMAL_SKIP_CONNECT_INDEX
from genotypes import Genotype
from art.metrics import clever_u,clever_t,clever
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from tensorboardX import SummaryWriter
from thop import profile, clever_format
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"   # batchsize
# os.environ["CUDA_VISIBLE_DEVICES"]="1"   # batchsize
import math

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
# parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size')
# parser.add_argument('--batch_size', type=int, default=12, help='batch size')
# parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate', type=float, default=0.04, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
# parser.add_argument('--epochs', type=int, default=2, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP/checkpoints/', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
# parser.add_argument('--train_portion', type=float, default=0.01, help='portion of training data')
# parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
# parser.add_argument('--arch_learning_rate', type=float, default=5e-3, help='learning rate for arch encoding')
# parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_learning_rate', type=float, default=6e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
# parser.add_argument('--arch_weight_decay', type=float, default=0, help='weight decay for arch encoding')
parser.add_argument('--tmp_data_dir', type=str, default='data/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', action='append', default=[], help='dropout rate of skip connect')
parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
parser.add_argument('--add_layers', action='append', default=['0'], help='add layers')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')

args = parser.parse_args()

args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'

if len(args.add_width) == 3:
    add_width = args.add_width
else:
    add_width = [0, 0, 0]
if len(args.add_layers) == 3:
    add_layers = args.add_layers
else:
    add_layers = [0, 6, 12]
if len(args.dropout_rate) ==3:
    drop_rate = args.dropout_rate
else:
    drop_rate = [0.0, 0.0, 0.0]

# To be moved to args
num_to_keep = [5, 3, 1]
# num_to_drop = [3, 2, 2]
normal_num_to_drop = [4, 3, 2]
# normal_num_to_drop = [3, 2, 2]
reduce_num_to_drop = [2, 2, 1]
# handler = SummaryWriter(log_dir=args.save)
normal_max_writer = []
reduce_max_writer = []
best_normal_writer = []
best_reduce_writer = []
normal_min_writer = []
reduce_min_writer = []
tb_index = [0, 0, 0]
best_prec1 = 0
max_arch_reward_writer = SummaryWriter(logdir='{}/tb/max_arch_reward'.format(args.save))
best_reward_arch_writer = SummaryWriter(logdir='{}/tb/best_reward_arch'.format(args.save))
avg_params_writer = SummaryWriter(logdir='{}/tb/avg_params'.format(args.save))
for i in range(14):
    # normal_writer.append(tf.summary.FileWriter(logdir='{}/tb/normal_{}'.format(args.save, i)))
    normal_max_writer.append(SummaryWriter(logdir='{}/tb/normal_max_{}'.format(args.save, i)))
    reduce_max_writer.append(SummaryWriter(logdir='{}/tb/reduce_max_{}'.format(args.save, i)))
    best_normal_writer.append(SummaryWriter(logdir='{}/tb/best_normal_{}'.format(args.save, i)))
    best_reduce_writer.append(SummaryWriter(logdir='{}/tb/best_reduce_{}'.format(args.save, i)))

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    #  prepare dataset
    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    # num_train = int(len(train_data)*0.2)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers)


    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    normal = []
    reduce = []
    for i in range(14):
        normal.append([True for j in range(len(PRIMITIVES_NORMAL))])
    for i in range(14):
        reduce.append([True for j in range(len(PRIMITIVES_REDUCE))])
    switches_normal = copy.deepcopy(normal)
    switches_reduce = copy.deepcopy(reduce)

    #switches_normal = [[True, True, True, False, True, True, False, True, False, False], [False, True, True, False, False, True, False, True, True, True], [False, True, True, False, True, True, True, True, False, False], [False, True, True, False, False, False, True, True, True, True], [True, True, False, False, True, True, True, True, False, False], [True, True, True, False, True, True, False, True, False, False], [True, True, False, True, True, False, False, True, True, False], [True, True, False, True, True, True, False, True, False, False], [True, False, True, True, True, True, True, False, False, False], [True, True, False, True, False, True, False, True, False, True], [False, True, False, True, False, True, False, True, True, True], [True, True, True, False, False, True, True, True, False, False], [True, False, True, True, True, True, True, False, False, False], [True, False, False, True, True, True, True, True, False, False]]
    #switches_reduce = [[True, False, True, False, True, True], [True, True, True, True, False, False], [True, False, True, False, True, True], [True, True, True, True, False, False], [False, True, True, True, True, False], [True, False, True, False, True, True], [False, False, True, True, True, True], [False, True, True, True, True, False], [False, True, True, True, True, False], [True, False, True, False, True, True], [True, True, True, True, False, False], [False, True, True, True, True, False], [False, False, True, True, True, True], [True, False, True, True, True, False]]
    # eps_no_archs = [20, 20, 20]
    # eps_no_archs = [5, 5, 5]
    eps_no_archs = [15, 15, 15]
    # eps_no_archs = [1, 1, 1]
    # eps_no_archs = [0, 0, 0]
    for sp in range(len(num_to_keep)):
        # if sp < 2:
        #     continue
        normal_min_writer.clear()
        reduce_min_writer.clear()
        for i in range(14):
            normal_min_writer_per = []
            for j in range(normal_num_to_drop[sp]):
                normal_min_writer_per.append(SummaryWriter(logdir='{}/tb/normal_min_{}_{}'.format(args.save, i, j)))
            normal_min_writer.append(normal_min_writer_per)

            reduce_min_writer_per = []
            for j in range(reduce_num_to_drop[sp]):
                reduce_min_writer_per.append(SummaryWriter(logdir='{}/tb/reduce_min_{}_{}'.format(args.save, i, j)))
            reduce_min_writer.append(reduce_min_writer_per)

        model = Network(args.init_channels + int(add_width[sp]), CIFAR_CLASSES, args.layers + int(add_layers[sp]), \
                        criterion, switches_normal=switches_normal, switches_reduce=switches_reduce, p=float(drop_rate[sp]))
        model = nn.DataParallel(model)
        model = model.cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        network_params = []
        for k, v in model.named_parameters():
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                network_params.append(v)       
        optimizer = torch.optim.SGD(
                network_params,
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                    lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        # optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
        #                                lr=args.arch_learning_rate, betas=(0, 0.999), weight_decay=args.arch_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2
        best_reward = 0
        # cur_sub_model = get_cur_model(model,switches_normal,switches_reduce,num_to_keep,num_to_drop,sp)
        for epoch in range(epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()
            # training
            if epoch < eps_no_arch:
            # if 0:
                model.module.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                model.module.update_p()
                train_acc, train_obj = train(sp,train_queue, valid_queue, model, network_params, criterion, optimizer, \
                                             optimizer_a, switches_normal, switches_reduce, training_arch = False)
            else:
                model.module.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor) 
                model.module.update_p()
                train_acc, train_obj = train(sp,train_queue, valid_queue, model, network_params, criterion, optimizer, \
                                             optimizer_a, switches_normal, switches_reduce, training_arch = True)
                # if epoch % 3 == 0:
                # if 1:
                #     train_arch(valid_queue,model,optimizer_a)
            logging.info('Train_acc %f', train_acc)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            # validation
            if epochs - epoch < 5:
            # if 1:
                valid_acc, valid_obj = infer(valid_queue, model, criterion)
                logging.info('Valid_acc %f', valid_acc)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
        print('------Dropping %d paths------' % normal_num_to_drop[sp])
        print('------Dropping %d paths------' % reduce_num_to_drop[sp])
        # Save switches info for s-c refinement.
        if sp == len(num_to_keep) - 1:
            switches_normal_2 = copy.deepcopy(switches_normal)
            switches_reduce_2 = copy.deepcopy(switches_reduce)
        # drop operations with low architecture weights
        arch_param = model.module.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()        
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES_NORMAL)):
                if switches_normal[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                # for the last stage, drop all Zero operations
                drop = get_min_k_no_zero(normal_prob[i, :], idxs, normal_num_to_drop[sp])
            else:
                drop = get_min_k(normal_prob[i, :], normal_num_to_drop[sp])
            for idx in drop:
                switches_normal[i][idxs[idx]] = False
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES_REDUCE)):
                if switches_reduce[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(reduce_prob[i, :], idxs, reduce_num_to_drop[sp])
            else:
                drop = get_min_k(reduce_prob[i, :], reduce_num_to_drop[sp])
            for idx in drop:
                switches_reduce[i][idxs[idx]] = False
        logging.info('switches_normal = %s', switches_normal)
        logging_switches(switches_normal, reduction = False)
        logging.info('switches_reduce = %s', switches_reduce)
        logging_switches(switches_reduce, reduction = True)
        
        if sp == len(num_to_keep) - 1:
            arch_param = model.module.arch_parameters()
            normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
            reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
            normal_final = [0 for idx in range(14)]
            reduce_final = [0 for idx in range(14)]
            # remove all Zero operations
            for i in range(14):
                if switches_normal_2[i][0] == True:
                    normal_prob[i][0] = 0
                normal_final[i] = max(normal_prob[i])
                if switches_reduce_2[i][0] == True:
                    reduce_prob[i][0] = 0
                reduce_final[i] = max(reduce_prob[i])                
            # Generate Architecture, similar to DARTS
            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            for i in range(3):  # 选出最大的两个前序节点
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1
            # set switches according the ranking of arch parameters
            for i in range(14):
                if not i in keep_normal:
                    for j in range(len(PRIMITIVES_NORMAL)):
                        switches_normal[i][j] = False
                if not i in keep_reduce:
                    for j in range(len(PRIMITIVES_REDUCE)):
                        switches_reduce[i][j] = False
            # translate switches into genotypep
            genotype = parse_network(switches_normal, switches_reduce)
            logging.info(genotype)
            ## restrict skipconnect (normal cell only)
            logging.info('Restricting skipconnect...')
            # generating genotypes with different numbers of skip-connect operations
            for sks in range(0, 9):
                max_sk = 8 - sks                
                num_sk = check_sk_number(switches_normal)               
                if not num_sk > max_sk:
                    continue
                while num_sk > max_sk:
                    normal_prob = delete_min_sk_prob(switches_normal, switches_normal_2, normal_prob)
                    switches_normal = keep_1_on(switches_normal_2, normal_prob, reduction = False)
                    switches_normal = keep_2_branches(switches_normal, normal_prob)
                    num_sk = check_sk_number(switches_normal)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype = parse_network(switches_normal, switches_reduce)
                logging.info(genotype)              

def get_cur_model(model, cur_switches_normal, cur_switches_reduce):
    sm_dim = -1

    switches_normal = [[False for col in range(len(PRIMITIVES_NORMAL))] for row in range(len(model.module.switches_normal))]
    switches_reduce = [[False for col in range(len(PRIMITIVES_REDUCE))] for row in range(len(model.module.switches_reduce))]

    normal_sel_index, reduce_sel_index = model.module.set_log_prob()
    #logging.info(normal_sel_index)
    #logging.info(reduce_sel_index)
    for i in range(14):
        idxs = []
        for j in range(len(PRIMITIVES_NORMAL)):
            if cur_switches_normal[i][j]:
                idxs.append(j)
        #logging.info(idxs)
        switches_normal[i][idxs[normal_sel_index[i]]] = True
    # for i,idx in enumerate(reduce_sel_index):   # 采样 需要挪到train 里面去
    #     for j in range(model.module.switch_reduce_on):
    #         if j != idx:
    #             switches_reduce[i][j] = False
    for i in range(14):
        idxs = []
        for j in range(len(PRIMITIVES_REDUCE)):
            if cur_switches_reduce[i][j]:
                idxs.append(j)
        switches_reduce[i][idxs[reduce_sel_index[i]]] = True

    # model.module.set_sub_net(switches_normal, switches_reduce)
    model.module.set_sub_net(normal_sel_index, reduce_sel_index)
    # switches_normal = [[False, False, False, False, False, False, False, False, False, True], [False, False, False, False, False, False, False, False, True, False], [False, False, False, False, False, False, False, True, False, False], [False, False, False, False, False, False, False, False, True, False], [False, False, False, False, False, True, False, False, False, False], [False, False, False, False, False, False, False, True, False, False], [False, False, False, False, False, False, False, False, True, False], [False, False, False, False, False, True, False, False, False, False], [False, False, False, True, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, True, False], [False, False, False, False, False, False, False, False, True, False], [False, False, False, False, False, False, False, False, True, False], [False, False, False, True, False, False, False, False, False, False], [False, False, True, False, False, False, False, False, False, False]]
    # with none
    # switches_normal = [[False, False, False, False, False, False, False, False, False, True], [False, False, False, False, False, False, False, False, True, False], [False, False, False, False, False, False, False, True, False, False], [False, False, False, False, False, False, False, False, True, False], [None, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, True, False, False], [False, False, False, False, False, False, False, False, True, False], [False, False, False, False, False, True, False, False, False, False], [False, False, False, True, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, True, False], [False, False, False, False, False, False, False, False, True, False], [False, False, False, False, False, False, False, False, True, False], [True, False, False, False, False, False, False, False, False, False], [True, False, False, False, False, False, False, False, False, False]]
    # switches_reduce = [[False, False, True, False, False, False], [False, False, True, False, False, False], [False, False, True, False, False, False], [False, False, True, False, False, False], [False, False, False, True, False, False], [False, False, False, False, True, False], [False, False, True, False, False, False], [False, True, False, False, False, False], [False, False, True, False, False, False], [False, False, True, False, False, False], [False, True, False, False, False, False], [False, False, True, False, False, False], [False, False, True, False, False, False], [False, False, False, False, True, False]]

    genotype =parse_network(switches_normal, switches_reduce)
    return normal_sel_index, reduce_sel_index, genotype

def set_max_model(model, cur_switches_normal, cur_switches_reduce):
    sm_dim = -1

    switches_normal = [[False for col in range(len(PRIMITIVES_NORMAL))] for row in range(len(model.module.switches_normal))]
    switches_reduce = [[False for col in range(len(PRIMITIVES_REDUCE))] for row in range(len(model.module.switches_reduce))]

    arch_param = model.module.arch_parameters()
    normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
    reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
    normal_sel_index = np.argmax(normal_prob, 1)
    reduce_sel_index = np.argmax(reduce_prob, 1)

    # remove all Zero operations
    # for i,idx in enumerate(normal_sel_index):   # 采样 需要挪到train 里面去
    #     for j in range(model.module.switch_normal_on):
    #         if j != idx:
    #             switches_normal[i][j] = False
    for i in range(14):
        idxs = []
        for j in range(len(PRIMITIVES_NORMAL)):
            if cur_switches_normal[i][j]:
                idxs.append(j)
        switches_normal[i][idxs[normal_sel_index[i]]] = True
    # for i,idx in enumerate(reduce_sel_index):   # 采样 需要挪到train 里面去
    #     for j in range(model.module.switch_reduce_on):
    #         if j != idx:
    #             switches_reduce[i][j] = False
    for i in range(14):
        idxs = []
        for j in range(len(PRIMITIVES_REDUCE)):
            if cur_switches_reduce[i][j]:
                idxs.append(j)
        switches_reduce[i][idxs[reduce_sel_index[i]]] = True

    # model.module.set_sub_net(switches_normal, switches_reduce)
    model.module.set_sub_net(normal_sel_index, reduce_sel_index)
    # genotype =parse_network(switches_normal, switches_reduce)
    return normal_sel_index, reduce_sel_index#, genotype


tb_index = [0, 0, 0]
R_L1 = 40
R_L2 = 2
R_LI = 0.1

best_normal_indices = []
best_reduce_indices = []
def train_arch(stage, step, valid_queue, model, optimizer_a, cur_switches_normal, cur_switches_reduce ):
    global best_prec1
    global best_normal_indices
    global best_reduce_indices
    # for step in range(100):
    try:
        input_search, target_search = next(valid_queue_iter)
    except:
        valid_queue_iter = iter(valid_queue)
        input_search, target_search = next(valid_queue_iter)
    input_search = input_search.cuda()
    target_search = target_search.cuda(non_blocking=True)
    normal_grad_buffer = []
    reduce_grad_buffer = []
    reward_buffer = []
    params_buffer = []
    flops_list = []
    params_list = []
    # cifar_mu = np.ones((3, 32, 32))
    # cifar_mu[0, :, :] = 0.4914
    # cifar_mu[1, :, :] = 0.4822
    # cifar_mu[2, :, :] = 0.4465

# (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # cifar_std = np.ones((3, 32, 32))
    # cifar_std[0, :, :] = 0.2471
    # cifar_std[1, :, :] = 0.2435
    # cifar_std[2, :, :] = 0.2616
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # classifier = PyTorchClassifier(
    #     model=model,
    #     clip_values=(0.0, 1.0),
    #     preprocessing=(cifar_mu, cifar_std),
    #     loss=criterion,
    #     optimizer=optimizer,
    #     input_shape=(3, 32, 32),
    #     nb_classes=10,
    # )

    for batch_idx in range(model.module.rl_batch_size): # 多采集几个网络，测试
        # sample the submodel
        # if stage == 1:
        #     print("ok")
        normal_indices, reduce_indices, genotype = get_cur_model(model, cur_switches_normal, cur_switches_reduce)
        # return 0.0, 0.0
        # attack = FastGradientMethod(estimator=model, eps=0.2)
        # x_test_adv = attack.generate(x=x_test)
        # res = clever_u(classifier,valid_queue.dataset.data[-1].transpose(2,0,1) , 2, 2, R_LI, norm=np.inf, pool_factor=3)
        # print(res)
        # validat the sub_model
        with torch.no_grad():
            logits= model(input_search)
            prec1, _ = utils.accuracy(logits, target_search, topk=(1,5))
        sub_model = NetworkCIFAR(36, CIFAR_CLASSES, 20, False, genotype)
        sub_model.drop_path_prob = 0
        # para0 = utils.count_parameters_in_MB(sub_model)
        input = torch.randn(1,3,32,32)
        flops, params = profile(sub_model, inputs = (input,), )
        flops_s, params_s = clever_format([flops, params], "%.3f")
        flops, params = flops/1e9, params/1e6
        params_buffer.append(params)
        flops_list.append(flops_s)
        params_list.append(params_s)


            # prec1 = np.random.rand()
        if model.module._arch_parameters[0].grad is not None:
            model.module._arch_parameters[0].grad.data.zero_()
        if model.module._arch_parameters[1].grad is not None:
            model.module._arch_parameters[1].grad.data.zero_()
        obj_term = 0
        for i in range(14):
            obj_term = obj_term + model.module.normal_log_prob[i]
            obj_term = obj_term + model.module.reduce_log_prob[i]
        loss_term = -obj_term
        # backward
        loss_term.backward()
        # take out gradient dict
        normal_grad_buffer.append(model.module._arch_parameters[0].grad.data.clone())
        reduce_grad_buffer.append(model.module._arch_parameters[1].grad.data.clone())
        reward = calculate_reward(stage, prec1, params)
        reward_buffer.append(reward)
        # recode best_reward index
        if prec1 > best_prec1:
            best_prec1 = prec1
            best_normal_indices = normal_indices
            best_reduce_indices = reduce_indices
        # else:
        #     best_normal_indices = []
        #     best_reduce_indices = []
    logging.info(flops_list)
    logging.info(params_list)
    logging.info(normal_indices.detach().cpu().numpy().squeeze())
    logging.info(reduce_indices.detach().cpu().numpy().squeeze())
    logging.info(genotype)
    avg_reward = sum(reward_buffer) / model.module.rl_batch_size
    avg_params = sum(params_buffer) / model.module.rl_batch_size
    if model.module.baseline == 0:
        model.module.baseline = avg_reward
    else:
        model.module.baseline += model.module.baseline_decay_weight * (avg_reward - model.module.baseline) # hs
        # model.module.baseline = model.module.baseline_decay_weight * model.module.baseline + \
        #                         (1-model.module.baseline_decay_weight) * avg_reward

    model.module._arch_parameters[0].grad.data.zero_()
    model.module._arch_parameters[1].grad.data.zero_()
    for j in range(model.module.rl_batch_size):
        model.module._arch_parameters[0].grad.data += (reward_buffer[j] - model.module.baseline) * normal_grad_buffer[j]
        model.module._arch_parameters[1].grad.data += (reward_buffer[j] - model.module.baseline) * reduce_grad_buffer[j]
    model.module._arch_parameters[0].grad.data /= model.module.rl_batch_size
    model.module._arch_parameters[1].grad.data /= model.module.rl_batch_size
    # if step % 50 == 0:
    #     logging.info(model.module._arch_parameters[0].grad.data)
    #     logging.info(model.module._arch_parameters[0])
    # apply gradients
    nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
    optimizer_a.step()

    if step % args.report_freq == 0:
        #     logging.info(model.module._arch_parameters[0])
        # valid the argmax arch
        logging.info('REINFORCE [step %d]\t\tMean Reward %.4f\tBaseline %.4f\tBest Sampled Prec1 %.4f', step, avg_reward, model.module.baseline, best_prec1)
        max_normal_index, max_reduce_index = set_max_model(model, cur_switches_normal, cur_switches_reduce)
        logits= model(input_search)
        prec1, _ = utils.accuracy(logits, target_search, topk=(1,5))
        logging.info('REINFORCE [step %d]\t\tCurrent Max Architecture Reward %.4f\t\tAvarage Params %.3f', step, prec1/100, avg_params)
        max_arch_reward_writer.add_scalar('max_arch_reward_{}'.format(stage), prec1, tb_index[stage])
        avg_params_writer.add_scalar('avg_params_{}'.format(stage), avg_params, tb_index[stage])
        logging.info(max_normal_index)
        logging.info(max_reduce_index)
        best_reward_arch_writer.add_scalar('best_prec1_arch_{}'.format(stage), best_prec1, tb_index[stage])

        logging.info(np.around(torch.Tensor(reward_buffer).numpy(),3))
        # logging.info(model.module.normal_probs)
        # logging.info(model.module.reduce_probs)
        logging.info(model.module.alphas_normal)
        logging.info(model.module.alphas_reduce)

        for i in range(14):
            normal_max_writer[i].add_scalar('normal_max_arch_{}'.format(stage), np.argmax(model.module.normal_probs.detach().cpu()[i].numpy()), tb_index[stage])

            normal_min_k = get_min_k(model.module.normal_probs.detach().cpu()[i].numpy(), normal_num_to_drop[stage])
            for j in range(normal_num_to_drop[stage]):
                normal_min_writer[i][j].add_scalar('normal_min_arch_{}_{}'.format(stage, j), normal_min_k[j], tb_index[stage])

            best_normal_writer[i].add_scalar('best_normal_index_{}'.format(stage), best_normal_indices[i].cpu().numpy(), tb_index[stage])

        for i in range(14):
            reduce_max_writer[i].add_scalar('reduce_max_arch_{}'.format(stage), np.argmax(model.module.reduce_probs.detach().cpu()[i].numpy()), tb_index[stage])

            reduce_min_k = get_min_k(model.module.reduce_probs.detach().cpu()[i].numpy(), reduce_num_to_drop[stage])
            for j in range(reduce_num_to_drop[stage]):
                reduce_min_writer[i][j].add_scalar('reduce_min_arch_{}_{}'.format(stage, j), reduce_min_k[j], tb_index[stage])

            best_reduce_writer[i].add_scalar('best_reduce_index_{}'.format(stage), best_reduce_indices[i].cpu().numpy(), tb_index[stage])

        best_prec1 = 0
        tb_index[stage]+=1

    model.module.restore_super_net()

def train(stage,train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, cur_switches_normal, cur_switches_reduce, training_arch = False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    # global baseline
    for step, (input, target) in enumerate(train_queue):
        model.train()
        if training_arch:
            if step % model.module.rl_interval_steps == 0:
                train_arch(stage, step, valid_queue, model, optimizer_a, cur_switches_normal, cur_switches_reduce)

        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # para1 = utils.count_parameters_in_MB(model)
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(network_params, args.grad_clip)
        # if not training_arch:
        optimizer.step()  # 去掉step 后， valid 下降的现象仍然存在

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            # logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, loss.data.item(), prec1.data.item(), prec5.data.item())
            # model.eval()  # 同样的输入input， 在REINFORCE的基础上，加上eval()后，计算所得的prec1 差别就非常大
            # with torch.no_grad():
            #     logits = model(input)
            # prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            # logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, loss.data.item(), prec1.data.item(), prec5.data.item())
            # logging.info(model.module._arch_parameters[0])
        # if training_arch:
        #     infer(valid_queue,model,criterion)
    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    # model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

a = 1
# b = 0.0
# c = 1
P_base = [6.4, 5.4, 4.4]
# D_base = 0.8
# F_base = 1.4
# alpha = -0.2
# alpha = -0.3
# alpha = [-0.45, -0.10, -0.10]
alpha = [-0.45, -0.25, -0.35]
# beta = -0.4
# gamma = -0.6

def calculate_reward(stage, prec1, params):

    mo_params_coe = a * math.pow(params/P_base[stage], alpha[stage])   #params M
    # mo_delay_coe = b * math.pow(delay/D_base, beta)  #inference delay S
    # mo_flops_coe = c * math.pow(flops/F_base, gamma)    # flops G
    # reward = prec1 * ( mo_params_coe +  mo_delay_coe + mo_flops_coe)
    reward = prec1 * ( mo_params_coe )
    return reward

def parse_network(switches_normal, switches_reduce):

    def _parse_switches(switches, reduction):
        if reduction:
            PRIMITIVES = PRIMITIVES_REDUCE
        else:
            PRIMITIVES = PRIMITIVES_NORMAL
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene
    gene_normal = _parse_switches(switches_normal, reduction = False)
    gene_reduce = _parse_switches(switches_reduce, reduction = True)
    
    concat = range(2, 6)
    
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat, 
        reduce=gene_reduce, reduce_concat=concat
    )
    
    return genotype

def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1
    
    return index
def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True 
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index
        
def logging_switches(switches, reduction = False):
    if reduction:
        PRIMITIVES = PRIMITIVES_REDUCE
    else:
        PRIMITIVES = PRIMITIVES_NORMAL
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)
        
def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        # if switches[i][3]:
        if switches[i][NORMAL_SKIP_CONNECT_INDEX]:
            count = count + 1
    
    return count

def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][NORMAL_SKIP_CONNECT_INDEX]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx
    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0
    
    return probs_out

def keep_1_on(switches_in, probs, reduction = False):
    switches = copy.deepcopy(switches_in)
    if reduction:
        PRIMITIVES = PRIMITIVES_REDUCE
    else:
        PRIMITIVES = PRIMITIVES_NORMAL
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 2)
        for idx in drop:
            switches[i][idxs[idx]] = False            
    return switches

def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES_NORMAL)):
                switches[i][j] = False  
    return switches  

if __name__ == '__main__':
    # start_time = time.time()
    # main()
    # end_time = time.time()
    # duration = end_time - start_time
    # logging.info('Total searching time: %ds', duration)
    switches_normal = [[False, False, True, False, False, False, False, False, False, False], [False, True, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, True, False, False], [False, False, False, False, False, False, False, True, False, False], [False, False, False, False, False, True, False, False, False, False], [False, False, False, False, False, False, False, True, False, False], [False, True, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, True, False, False], [False, False, False, False, False, False, False, True, False, False], [False, True, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, True, False, False], [False, False, False, False, False, True, False, False, False, False], [False, False, False, False, False, False, True, False, False, False], [False, False, False, False, False, False, False, True, False, False]]
    switches_reduce = [[False, False, True, False, False, False], [False, False, True, False, False, False], [False, False, False, False, False, True], [False, False, False, True, False, False], [False, False, False, True, False, False], [False, False, False, False, False, True], [False, False, False, True, False, False], [False, False, True, False, False, False], [False, False, False, False, True, False], [False, False, False, False, True, False], [False, False, True, False, False, False], [False, False, False, True, False, False], [False, False, True, False, False, False], [False, False, False, False, True, False]]
    genotype = parse_network(switches_normal, switches_reduce)
    logging.info(genotype)