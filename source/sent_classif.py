#!/usr/bin/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Simple MLP classifier for sentence embeddings


import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils


################################################

def LoadData(bdir, dfn, lfn, dim=1024, bsize=32, shuffle=False, quiet=False):
    x = np.fromfile(bdir + dfn, dtype=np.float32, count=-1)
    x.resize(x.shape[0] // dim, dim)

    lbl = np.loadtxt(bdir + lfn, dtype=np.int32)
    lbl.reshape(lbl.shape[0], 1)
    if not quiet:
        print(' - read {:d}x{:d} elements in {:s}'.format(x.shape[0], x.shape[1], dfn))
        print(' - read {:d} labels [{:d},{:d}] in {:s}'
              .format(lbl.shape[0], lbl.min(), lbl.max(), lfn))

    D = data_utils.TensorDataset(torch.from_numpy(x), torch.from_numpy(lbl))
    loader = data_utils.DataLoader(D, batch_size=bsize, shuffle=shuffle)
    return loader


################################################

class Net(nn.Module):
    def __init__(self, idim=1024, odim=2, nhid=None,
                 dropout=0.0, gpu=0, activation='TANH'):
        super(Net, self).__init__()
        self.gpu = gpu
        modules = []

        modules = []
        print(' - mlp {:d}'.format(idim), end='')
        if len(nhid) > 0:
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
            nprev = idim
            for nh in nhid:
                if nh > 0:
                    modules.append(nn.Linear(nprev, nh))
                    nprev = nh
                    if activation == 'TANH':
                        modules.append(nn.Tanh())
                        print('-{:d}t'.format(nh), end='')
                    elif activation == 'RELU':
                        modules.append(nn.ReLU())
                        print('-{:d}r'.format(nh), end='')
                    else:
                       raise Exception('Unrecognized activation {activation}')
                    if dropout > 0:
                        modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(nprev, odim))
            print('-{:d}, dropout={:.1f}'.format(odim, dropout))
        else:
            modules.append(nn.Linear(idim, odim))
            print(' - mlp %d-%d'.format(idim, odim))
        self.mlp = nn.Sequential(*modules)
        # Softmax is included CrossEntropyLoss !

        if self.gpu >= 0:
            self.mlp = self.mlp.cuda()

    def forward(self, x):
        return self.mlp(x)

    def TestCorpus(self, dset, name=' Dev', nlbl=4):
        correct = 0
        total = 0
        all_predicted = torch.Tensor().long()
        self.mlp.train(mode=False)
        corr = np.zeros(nlbl, dtype=np.int32)

        # iterate through batches - dset is a DataLoader, data is a 2-element list. 
        # First element is a (12,1024) tensor (12 batches each with sentence embedding of 1024 length)
        # Second element is a (12,1) tensor with corresponding labels
        for data in dset:
            X, Y = data
            Y = Y.long()
            if self.gpu >= 0:
                X = X.cuda()
                Y = Y.cuda()
            outputs = self.mlp(X)
            _, predicted = torch.max(outputs.data, 1)
            
            total += Y.size(0)
            correct += (predicted == Y).int().sum()
            for i in range(nlbl):
                corr[i] += (predicted == i).int().sum()

            # add batch predictions to previous predictions
            all_predicted = torch.cat((all_predicted, predicted.cpu()), 0)

        print(' | {:4s}: {:5.2f}%'
                         .format(name, 100.0 * correct.float() / total), end='')
        # TODO uncomment later, for now i don't want to see classes
        # print(' | classes:', end='')
        # for i in range(nlbl):
        #     print(' {:5.2f}'.format(100.0 * corr[i] / total), end='')

        return correct, total, all_predicted

    def get_labels(self, dset):
        all_predicted = torch.Tensor().long()
        all_y = torch.Tensor().long()
        self.mlp.train(mode=False)

        # iterate through batches - dset is a DataLoader, data is a 2-element list. 
        # First element is a (12,1024) tensor (12 batches each with sentence embedding of 1024 length)
        # Second element is a (12,1) tensor with corresponding labels
        for data in dset:
            X, Y = data
            Y = Y.long()
            if self.gpu >= 0:
                X = X.cuda()
                Y = Y.cuda()
            outputs = self.mlp(X)
            _, predicted = torch.max(outputs.data, 1)
            

            # add batch predictions to previous predictions
            all_predicted = torch.cat((all_predicted, predicted.cpu()), 0)
            all_y = torch.cat((all_y, Y.cpu()), 0)

        return all_y, all_predicted


################################################

parser = argparse.ArgumentParser(
           formatter_class=argparse.RawDescriptionHelpFormatter,
           description="Simple sentence classifier")

# Data
parser.add_argument(
    '--base-dir', '-b', type=str, required=True, metavar='PATH',
    help="Directory with all the data files)")
parser.add_argument(
    '--save', '-s', type=str, required=False, metavar='PATH', default="",
    help="File in which to save best network")
parser.add_argument(
    '--train', '-t', type=str, required=True, metavar='STR',
    help="Name of training corpus")
parser.add_argument(
    '--train-labels', '-T', type=str, required=True, metavar='STR',
    help="Name of training corpus (labels)")
parser.add_argument(
    '--dev', '-d', type=str, required=True, metavar='STR',
    help="Name of development corpus")
parser.add_argument(
    '--dev-labels', '-D', type=str, required=True, metavar='STR',
    help="Name of development corpus (labels)")
parser.add_argument(
    '--test', '-e', type=str, required=True, metavar='STR',
    help="Name of test corpus without language extension")
parser.add_argument(
    '--test-labels', '-E', type=str, required=True, metavar='STR',
    help="Name of test corpus without language extension (labels)")
parser.add_argument(
    '--lang', '-L', nargs='+', default=None, 
    help="List of languages to test on")

# network definition
parser.add_argument(
    "--dim", "-m", type=int, default=1024,
    help="Dimension of sentence embeddings")
parser.add_argument(
    '--nhid', '-n', type=int, default=[0], nargs='+',
    help="List of hidden layer(s) dimensions")
parser.add_argument(
    "--nb-classes", "-c", type=int, default=2,
    help="Number of output classes")
parser.add_argument(
    '--dropout', '-o', type=float, default=0.0, metavar='FLOAT',
    help="Value  of dropout")
parser.add_argument(
    '--nepoch', '-N', type=int, default=100, metavar='INT',
    help="Number of epochs")
parser.add_argument(
    '--bsize', '-B', type=int, default=128, metavar='INT',
    help="Batch size")
parser.add_argument(
    '--seed', '-S', type=int, default=123456789, metavar='INT',
    help="Initial random seed")
parser.add_argument(
    '--lr', type=float, default=0.001, metavar='FLOAT',
    help='Learning rate')
parser.add_argument(
    '--wdecay', type=float, default=0.0, metavar='FLOAT',
    help='Weight decay')
parser.add_argument(
    '--gpu', '-g', type=int, default=-1, metavar='INT',
    help="GPU id (-1 for CPU)")

parser.add_argument(
    '--create_labels', '-l', action='store_true',
    help="Whether to create labels (true) or just train and evaluate (false)")

args = parser.parse_args()

print(' - base directory: {}'.format(args.base_dir))
args.base_dir = args.base_dir + "/"

train_loader = LoadData(args.base_dir, args.train, args.train_labels,
                        dim=args.dim, bsize=args.bsize, shuffle=True)

dev_loader = LoadData(args.base_dir, args.dev, args.dev_labels,
                      dim=args.dim, bsize=args.bsize, shuffle=False)

# set GPU and random seed
torch.cuda.set_device(args.gpu)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(" - setting seed to %d" % args.seed)

# create network
net = Net(idim=args.dim, odim=args.nb_classes,
          nhid=args.nhid, dropout=args.dropout, gpu=args.gpu)
if args.gpu >= 0:
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()

#optimizer = optim.Adam(net.parameters(), weight_decay=0.0)
# default: pytorch/optim/adam.py
# Py0.4: lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
# Py1.0: lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
optimizer = optim.Adam(net.parameters(),
                       lr=args.lr,
                       weight_decay=args.wdecay,
                       betas=(0.9, 0.999),
                       eps=1e-8,
                       amsgrad=False)

corr_best = 0
# loop multiple times over the dataset
for epoch in range(args.nepoch):

    loss_epoch = 0.0
    print('Ep {:4d}'.format(epoch), end='')
    # for inputs, labels in train_loader:
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        labels = labels.long()
        if args.gpu >= 0:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        net.zero_grad()

        # forward + backward + optimize
        net.train(mode=True)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()

    print(' | loss {:e}'.format(loss_epoch), end='')
    # corr_train, nbex_train, _ = net.TestCorpus(train_loader, 'Train')
    corr, nbex, _ = net.TestCorpus(dev_loader, 'Dev')
    if corr >= corr_best:
        print(' | saved')
        corr_best = corr
        # corr_train_best = corr_train
        net_best = copy.deepcopy(net)
    else:
        print('')

if 'net_best' in globals():
    if args.save != '':
        torch.save(net_best.cpu(), args.save)
    print("# epochs: {}, lr: {}, nhid: {}, drop: {}, bsize: {}"
          .format(args.nepoch,args.lr, args.nhid, args.dropout, args.bsize))
    # print('Best dev - Dev {:5.2f}% Train {:5.2f}%'
    #       .format(100.0 * corr_best.float() / nbex, 100.0 * corr_train_best.float() / nbex_train))

    if args.gpu >= 0:
        net_best = net_best.cuda()
# TODO comment for finetuning because then we dont want to see the test set <- instead of uncommenting add a parameter and an if statement
# At first I thought we don't want to test when creating labels but later i decided that it's good to see the test metrics to make sure they are
# as good as expected. Maybe we should delete create_labels later.

    # if not args.create_labels:

        
    # else:
        # save ground truth and labels predicted on train and dev set
        # for l in args.lang:
        #     print("Language",l)
        #     for part in ['train','dev']:
        #         data_loader = LoadData(args.base_dir,f"{part}.enc." + l,
        #                             f"{part}.lbl." + l,
        #                             dim=args.dim, bsize=args.bsize,
        #                             shuffle=False, quiet=True)

        #         actuals, preds = net_best.get_labels(data_loader)
        #         # unsqueeze(1) changes tensor shape from (n) to (n,1)
        #         merge = torch.cat((actuals.unsqueeze(1), preds.unsqueeze(1)), 1)
        #         # multifit uses "ja" instead of "jp"
        #         if l=="jp":
        #             np.savetxt(f"preds-{part}-ja.csv",merge.numpy().astype(int),delimiter=",")
        #         else:
        #             np.savetxt(f"preds-{part}-{l}.csv",merge.numpy().astype(int),delimiter=",")

        # test on (several) languages
        for l in args.lang:
            test_loader = LoadData(args.base_dir, args.test + '.' + l,
                                args.test_labels + '.' + l,
                                dim=args.dim, bsize=args.bsize,
                                shuffle=False, quiet=True)
            print('Ep best | Eval Test lang {:s}'.format(l), end='')
            _, _, preds = net_best.TestCorpus(test_loader, 'Test')
            print('')



