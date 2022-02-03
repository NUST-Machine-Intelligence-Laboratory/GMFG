#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from torch.utils.data import DataLoader
import argparse
import pickle
from Imagefolder_modified import Imagefolder_modified
from Imagefolder_meta import Imagefolder_meta
from resnet_my import *
from utils import *
from PIL import ImageFile # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True

import time

class Manager(object):
    def __init__(self, args):
        print('------------------------------------------------------------------------------')
        print('Preparing the network and data ... ')
        self._options = args
        self._path = args.path
        os.popen('mkdir -p ' + self._path)
        self._data_base = args.data_base
        self._data_meta = args.data_meta
        self._class =args.n_classes
        self._drop_rate = args.drop_rate
        self._relabel_rate = args.relabel_rate
        self._tk = args.tk
        print('Basic information: ','data:', self._data_base,'  lr:', self._options.lr,'  w_decay:', self._options.w_decay)
        print('Parameter information: ', 'drop_rate:', self._drop_rate, 'relabel_rate:', self._relabel_rate, '  tk:',self._tk)
        print('------------------------------------------------------------------------------')
        # Network
        print('network:', self._options.net)
        if self._options.net == 'resnet18':
            NET = ResNet18_meta
            n = 512
        elif self._options.net == 'resnet34':
            NET = ResNet34_meta
            n = 512
        elif self._options.net == 'resnet50':
            NET = ResNet50_meta
            n = 2048
        else:
            raise AssertionError('Not implemented yet')

        self._NET = NET
        self._net = NET(n_classes=self._class, pretrained=True).cuda()

        # Criterion
        self._optimizer_a = torch.optim.SGD(self._net.params(), lr=self._options.lr, momentum=0.9, weight_decay=self._options.w_decay)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer_a, T_max=self._options.epochs)

        fnet = Feature_Net(n,256, 1)
        self._fnet = fnet.cuda()
        lr_f = self._options.lr
        self._optimizer_f = torch.optim.SGD(self._fnet.parameters(), lr=lr_f, momentum=0.9, weight_decay=self._options.w_decay)

        label_net = Meta_head(n, self._class)
        self._label_net = label_net.cuda()
        lr_label = self._options.lr
        self._optimizer_l = torch.optim.SGD(self._label_net.parameters(), lr=lr_label, momentum=0.9, weight_decay=self._options.w_decay)
        self._scheduler_l = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer_l,  T_max=self._options.epochs)

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # Load data
        self._test_data = Imagefolder_modified(os.path.join(self._data_base, 'val'), transform=test_transform, cached=False)
        self._meta_data = Imagefolder_modified(self._data_meta, transform=train_transform, cached=False)
        self._train_meta_data = Imagefolder_meta(os.path.join(self._data_base, 'train'), self._data_meta,
                                                 transform=train_transform, transform_meta= None, cached=False, number=self._options.number)

        print('number of classes in trainset is : {}'.format(len(self._train_meta_data.classes)))
        print('number of classes in testset is : {}'.format(len(self._test_data.classes)))

        self._meta_loader = DataLoader(self._meta_data, batch_size=self._options.batch_size,
                                        shuffle=True, num_workers=4, pin_memory=True)
        self._train_meta_loader = DataLoader(self._train_meta_data, batch_size=self._options.batch_size,
                                             shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = DataLoader(self._test_data, batch_size=16,
                                       shuffle=False, num_workers=4, pin_memory=True)

        self._wf = torch.zeros(len(self._train_meta_data))

    def _label_smoothing_cross_entropy(self,logit, label, epsilon=0.1, reduction='mean'):
        N = label.size(0)
        C = logit.size(1)
        smoothed_label = torch.full(size=(N, C), fill_value=epsilon / (C - 1))
        smoothed_label.scatter_(dim=1, index=torch.unsqueeze(label, dim=1).cpu(), value=1 - epsilon)

        if logit.is_cuda:
            smoothed_label = smoothed_label.cuda()

        log_logit = F.log_softmax(logit, dim=1)
        losses = -torch.sum(log_logit * smoothed_label, dim=1)  # (N)
        if reduction == 'none':
            return losses
        elif reduction == 'mean':
            return torch.sum(losses) / N
        elif reduction == 'sum':
            return torch.sum(losses)
        else:
            raise AssertionError('reduction has to be none, mean or sum')


    def _selection_loss_minibatch(self, logits, logits_l, labels, epoch):
        if epoch < self._tk or self._drop_rate == 0:
            loss = self._label_smoothing_cross_entropy(logits, labels, reduction = 'mean')
            return loss
        loss_all = F.cross_entropy(logits, labels, reduction='none')

        index_sorted = torch.argsort(loss_all.detach(), descending=False)
        num_remember = int((1 - self._drop_rate) * labels.size(0))
        index_clean = index_sorted[:num_remember]
        logits_final = logits[index_clean]
        labels_final = labels[index_clean]

        loss = self._label_smoothing_cross_entropy(logits_final, labels_final, reduction = 'mean')

        logits_l_final = logits_l[index_clean]
        loss += L_loss(logits_final, logits_l_final.detach())
        return loss

    def _relabel_loss_minibatch(self, logits, logits_l, labels, ids):
        loss_all = F.cross_entropy(logits, labels, reduction='none')

        index_sorted = torch.argsort(loss_all.detach(), descending=False)
        num_remember = int((1-self._drop_rate) * labels.size(0))
        #按照loss排序，分为clean（低loss）和noise（高loss）
        index_clean = index_sorted[:num_remember]
        index_noise = index_sorted[num_remember:]

        weight_feature = self._wf[ids[index_noise]]
        noise_sorted = torch.argsort(weight_feature.detach(), descending=True)
        num_relabel = int(self._relabel_rate * labels.size(0))
        index_relabel = index_noise[noise_sorted[:num_relabel]]
        labels_clean = labels[index_clean]

        loss = self._label_smoothing_cross_entropy(logits[index_clean], labels_clean, reduction='mean', epsilon=0.1)
        logits_final = logits[torch.cat((index_clean, index_relabel), 0)]
        logits_l_final = logits_l[torch.cat((index_clean, index_relabel), 0)]
        loss += L_loss(logits_final,logits_l_final.detach())
        return loss

    def train(self):
        """
        Train the network
        """
        print('Training ... ')
        best_accuracy = 0.0
        print('Epoch\tTrain Loss\tTrain Accuracy\tTest Accuracy\tLabel Accuracy\tEpoch Runtime')

        for t in range(self._options.epochs):
            epoch_start = time.time()
            epoch_loss = []
            num_correct = 0
            num_total = 0
            num_label=0

            for data in self._train_meta_loader:
                self._net.train(True)
                X, y, id, path, x_validation, y_validation = data

                # Data
                X = X.cuda()
                y = y.cuda()

                x_validation = x_validation.cuda()
                y_validation = y_validation.cuda()

                # Snet
                if t < self._tk  and self._relabel_rate > 0:
                    meta_net = self._NET(n_classes=self._class, pretrained=False).cuda()
                    meta_net.load_state_dict(self._net.state_dict())

                    y_f_hat, feature_c = meta_net(X)
                    cost = F.cross_entropy(y_f_hat, y, reduce=False)
                    cost_v = torch.reshape(cost, (len(cost), 1))

                    v_lambda = self._fnet(feature_c.detach())
                    norm_c = torch.sum(v_lambda)
                    # normalized
                    v_lambda_norm = v_lambda / norm_c

                    l_f_meta = torch.sum(cost_v * v_lambda_norm)

                    meta_net.zero_grad()
                    grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
                    meta_lr = self._optimizer_a.state_dict()['param_groups'][0]['lr']
                    meta_net.update_params(lr_inner=meta_lr, source_params=grads)  # Eq. 3
                    del grads
                    #
                    y_g_hat,_ = meta_net(x_validation)
                    l_g_meta = F.cross_entropy(y_g_hat, y_validation)
                    _, prediction = torch.max(y_g_hat.data, 1)

                    self._optimizer_f.zero_grad()
                    l_g_meta.backward()  # Eq. 4
                    self._optimizer_f.step()

                # meta head
                with torch.no_grad():
                    _, feature_c = self._net(x_validation)
                outputlabels = self._label_net(feature_c)
                loss_l = self._label_smoothing_cross_entropy(outputlabels, y_validation, reduction='mean')
                _ , prediction_labelnet = torch.max(outputlabels, 1)
                num_label += torch.sum(prediction_labelnet == y_validation.data).item()

                self._optimizer_l.zero_grad()
                loss_l.backward()
                self._optimizer_l.step()

                # Forward pass
                y_f, feature_c = self._net(X)
                with torch.no_grad():
                    y_l = self._label_net(feature_c)
                if t < self._tk and self._relabel_rate > 0:
                    with torch.no_grad():
                        w_f = self._fnet(feature_c)
                        self._wf[id] = w_f[:, 0].cpu().detach()

                if t >= self._tk and self._relabel_rate > 0:
                    loss = self._relabel_loss_minibatch(y_f,y_l,y,id)
                else:
                    loss = self._selection_loss_minibatch(y_f,y_l,y,t)

                _, prediction = torch.max(y_f, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()

                # Backward
                self._optimizer_a.zero_grad()
                loss.backward()
                self._optimizer_a.step()

                epoch_loss.append(loss.item())

            if self._options.plus:
                for data in self._meta_loader:
                    self._net.train(True)
                    X, y, _, _= data
                    # Data
                    X = X.cuda()
                    y = y.cuda()
                    y_f, _ = self._net(X)

                    loss = self._label_smoothing_cross_entropy(y_f,y, reduction='mean')

                    _, prediction = torch.max(y_f, 1)
                    num_total += y.size(0)
                    num_correct += torch.sum(prediction == y.data).item()

                    # Backward
                    self._optimizer_a.zero_grad()
                    loss.backward()
                    self._optimizer_a.step()

            # Record the test accuracy of each epoch
            test_accuracy = self.test(self._test_loader)
            train_accuracy = 100 * num_correct / num_total
            label_accuracy = 100 * num_label / len(self._train_meta_data)


            self._scheduler.step()  # the scheduler adjust lr based on test_accuracy
            self._scheduler_l.step()

            epoch_end = time.time()

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                print('*', end='')
                # Save mode
                torch.save(self._net.state_dict(), os.path.join(self._path, self._options.net + '.pth'))
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f' % (t + 1, sum(epoch_loss) / len(epoch_loss),
                                                            train_accuracy, test_accuracy, label_accuracy,
                                                            epoch_end - epoch_start))

        print('-----------------------------------------------------------------')

    def test(self, dataloader):
        """
        Compute the test accuracy

        Argument:
            dataloader  Test dataloader
        Return:
            Test accuracy in percentage
        """
        self._net.train(False) # set the mode to evaluation phase
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for X, y,_,_ in dataloader:
                # Data
                X = X.cuda()
                y = y.cuda()
                # Prediction
                score,_ = self._net(X)
                # score, _ = self._net(X)
                _, prediction = torch.max(score, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # set the mode to training phase
        return 100 * num_correct / num_total

    def test_categories(self, dataloader, model =None):
        self._net.train(False) # set the mode to evaluation phase
        if model!= None:
            model_dict = torch.load(model)
            self._net.load_state_dict(model_dict)

        num_correct = torch.zeros(self._class)
        num_total = torch.zeros(self._class)

        with torch.no_grad():
            for X, y,_,_ in dataloader:
                # Data
                X = X.cuda()
                y = y.cuda()

                score,_ = self._net(X)
                _, prediction = torch.max(score, 1)

                for i in range(y.size(0)):
                    if prediction[i] == y[i]:
                        num_correct[y[i]] +=1
                    num_total[y[i]]+=1

        result = 100 * num_correct / num_total
        self._net.train(True)  # set the mode to training phase
        return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
    parser.add_argument('--net', dest='net', type=str, default='resnet18',
                        help='supported options: resnet18, resnet50')
    parser.add_argument('--n_classes', dest='n_classes', type=int, default=200,
                        help='number of classes')
    parser.add_argument('--path', dest='path', type=str, default='model')
    parser.add_argument('--data_base', dest='data_base', type=str, default='/home/zcy/data/fg-web-data/web-bird')
    parser.add_argument('--data_meta', dest='data_meta', type=str, default='meta_data/bird')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-2)
    parser.add_argument('--w_decay', dest='w_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=80)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--drop_rate', type=float, default=0.35)
    parser.add_argument('--relabel_rate', type=float, default=0.05)
    parser.add_argument('--plus', action='store_true', help='Turns on training on validation set', default=False)
    parser.add_argument('--tk', type=int, default=5)
    parser.add_argument('--number', type=int, default=None)

    args = parser.parse_args()

    model = args.path

    print(os.path.join(os.popen('pwd').read().strip(), model))

    if not os.path.isdir(os.path.join(os.popen('pwd').read().strip(), model)):
        print('>>>>>> Creating directory \'model\' ... ')
        os.mkdir(os.path.join(os.popen('pwd').read().strip(), model))

    path = os.path.join(os.popen('pwd').read().strip(), model)

    manager = Manager(args)
    manager.train()