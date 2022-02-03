# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            # setattr(curr_mod, name, param)
            # print(curr_mod,name)
            # print(dir(curr_mod))
            # print(getattr(curr_mod,name))
            curr_mod.__dict__[name]=param
            # print(curr_mod.__getattr__(name))
            # tmp = curr_mod.__getattr__(name).data
            # curr_mod.__setattr__(name,nn.Parameter(param, requires_grad=True))
            # curr_mod.__dict__[name] = param
            # print(curr_mod.__getattr__(name).data - tmp)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    # def copy(self, other, same_var=False):
    #     for name, param in other.named_params():
    #         if not same_var:
    #             param = to_var(param.data.clone(), requires_grad=True)
    #         self.set_param(name, param)

class ResNet18_meta(MetaModule):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True, use_two_step=False):
        super().__init__()
        # print('| A ResNet18 network is instantiated, pre-trained: {}, '
              # 'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        # feature output is (N, 512)
        # self.features = nn.Sequential(*list(resnet18(pretrained=pretrained).children())[:-2])

        resnet = torchvision.models.resnet18(pretrained=self._pretrained)
        # feature output is (N, 512)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self._n_classes)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        feature_c = self.avgpool(x)
        feature_c = feature_c.view(N, -1)
        x = self.fc(feature_c)
        return x, feature_c

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class Feature_Net(nn.Module):
    def __init__(self, input, hidden, output):
        super(Feature_Net, self).__init__()
        print('MLP is built')
        self.linear1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

class Meta_head(nn.Module):
    def __init__(self, input, output):
        super(Meta_head, self).__init__()
        self.linear1 = nn.Linear(input, output)

    def forward(self, x):
        x = self.linear1(x)
        return x
        # return torch.sigmoid(x)

class ResNet34_meta(MetaModule):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True, use_two_step=True):
        super().__init__()
        # print('| A ResNet34 network is instantiated, pre-trained: {}, '
        #       'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes

        # self.features = nn.Sequential(*list(resnet34(pretrained=pretrained).children())[:-2])
        resnet = torchvision.models.resnet34(pretrained=self._pretrained)
        # feature output is (N, 512)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=self._n_classes)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        feature_c = self.avgpool(x)
        feature_c = feature_c.view(N, -1)
        x = self.fc(feature_c)
        return x, feature_c

class ResNet50(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True, use_two_step=False):
        super().__init__()
        print('| A ResNet50 network is instantiated, pre-trained: {}, '
              'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = torchvision.models.resnet50(pretrained=self._pretrained)
        # feature output is (N, 2048)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=self._n_classes)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = self.fc(x)
        return x

class ResNet50_meta(MetaModule):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True, use_two_step=False):
        super().__init__()
        # print('| A ResNet50 network is instantiated, pre-trained: {}, '
        #       'two-step-training: {}, number of classes: {}'.format(pretrained, use_two_step, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = torchvision.models.resnet50(pretrained=self._pretrained)
        # feature output is (N, 2048)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=self._n_classes)

        if self._pretrained:
            # Init the fc layer
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)
            if use_two_step:
                for params in self.features.parameters():
                    params.required_grad = False

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        feature = x.view(N, -1)
        x = self.fc(feature)
        return x, feature


if __name__ == '__main__':
    net = ResNet50()
    x = torch.rand(64, 3, 448, 448)
    y = net(x)
    print(y.shape)
