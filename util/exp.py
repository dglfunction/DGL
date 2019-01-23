from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

from functools import reduce

    
def weight_init(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_normal_(model.weight.data, nn.init.calculate_gain('tanh'))
        if model.bias is not None:
            nn.init.normal_(model.bias.data, mean=0, std=0.5)
    elif type(model) in [nn.Conv1d]:
        nn.init.kaiming_normal_(model.weight.data, nn.init.calculate_gain('tanh'))
        if model.bias is not None:
            model.bias.data.zero_()

class ActivatedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation=F.tanh, bn=False, p=0.):
        super(ActivatedLinear, self).__init__()
        self.module = nn.Linear(in_features, out_features, bias)
        self.activation=activation
        self.bn=None
        self.dropout = None
        if p > 0.:
            self.dropout = nn.Dropout(p)
        if bn:
            self.bn=torch.nn.BatchNorm1d(out_features)
            
        
    def forward(self, x):
        x = self.module(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class ActivatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=F.relu, bn=False, p=0.):
        super(ActivatedConv, self).__init__()
        self.module = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.activation=activation
        self.bn=None
        self.dropout = None
        if p > 0.:
            self.dropout = nn.Dropout(p)
        if bn:
            self.bn=torch.nn.BatchNorm2d(out_features)
            
        
    def forward(self, x):
        x = self.module(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.activation(x)
        return x
    
    
class FFNet(nn.Module):
    def __init__(self, n_inputs=10, n_classes=2, nonlinear=F.tanh, n_h_layers=5, h_h_layers=20, bn=False, p=0., bias=True,):
        super(FFNet, self).__init__()
        ## n_h_layers < 0 is only linear layer
        self.fc_layers=nn.ModuleList([ActivatedLinear(n_inputs, h_h_layers, activation=nonlinear, bn=bn, p=p, bias=bias,)])
        self.fc_layers.extend([ActivatedLinear(h_h_layers, h_h_layers, activation=nonlinear, bn=bn, p=p, bias=bias,) for i in range(n_h_layers)])
        self.fc_layers.append(nn.Linear(h_h_layers, n_classes, bias=bias))

        self.nonlinear=nonlinear        
        self.layers=nn.ModuleList(self.fc_layers.children())
    
    def forward(self, x):
        x = reduce(lambda x,f:f(x),self.fc_layers,x)
        return x
    
    def fc_layer_activations(self, layer_idx, x, layer_only=False):
        starting_layer = layer_idx if layer_only else 0
        x = reduce(lambda x, f:f(x), self.fc_layers[starting_layer:layer_idx+1], x)
        return x

def forward_layer(layer, x):
    try:
        layer_dimension = layer.module.weight.data.ndimension()
    except:
        layer_dimension = layer.weight.data.ndimension()
    if (layer_dimension == 2) and (x.ndimension() > 2):
        x = x.view(x.shape[0], -1)
    return layer(x)
    
    
class ConvFFNet(nn.Module):
    def __init__(
        self,
        n_inputs=10,
        n_classes=2,
        nonlinear=F.tanh,
        n_h_layers=5,
        h_h_layers=20,
        c_inputs=1,
        n_conv_layers=0,
        k_conv_layers=2,
        c_conv_layers=2,
        s_conv_layers=2,
        bn=False,
        p=0.,
        bias=True,
    ):
        super(ConvFFNet, self).__init__()
        
        if n_conv_layers==0:
            self.conv_layers=None
        else:
            self.conv_layers=nn.ModuleList(
                [
                    ActivatedConv(
                        c_inputs,
                        c_conv_layers,
                        k_conv_layers,
                        s_conv_layers,
                        activation=nonlinear,
                        bn=bn,
                        p=p,
                        bias=bias,
                    )
                ]
            )
            self.conv_layers.extend(
                [
                    ActivatedConv(
                        c_conv_layers*((c_conv_layers//c_inputs)**(i-1)),
                        c_conv_layers*((c_conv_layers//c_inputs)**i),
                        k_conv_layers,
                        s_conv_layers,
                        activation=nonlinear,
                        bn=bn,
                        p=p,
                        bias=bias,
                    ) 
                    for i in range(1, n_conv_layers)
                ]
            )
            # calculate n_inputs for the conv layer case
            side_input = n_inputs
            # (W−F+2P)/S+1
            for i in range(n_conv_layers):
                side_input = (side_input-k_conv_layers)//s_conv_layers + 1
            # chan size * num of channels    
            n_inputs  = (side_input**2) * c_conv_layers * ((c_conv_layers//c_inputs)**(n_conv_layers-1))
        
        self.fc_layers=nn.ModuleList([])    
        if n_h_layers >= 0:
            self.fc_layers.extend([ActivatedLinear(n_inputs, h_h_layers, activation=nonlinear, bn=bn, p=p, bias=bias,)])
            self.fc_layers.extend([ActivatedLinear(h_h_layers, h_h_layers, activation=nonlinear, bn=bn, p=p, bias=bias,) for i in range(n_h_layers)])
            
        self.fc_layers.append(nn.Linear(h_h_layers if n_h_layers >= 0 else n_inputs, n_classes, bias=bias))
        # joint list for retrain layer access
        self.layers=nn.ModuleList(self.conv_layers.children() if n_conv_layers > 0 else [])
        self.layers.extend(self.fc_layers.children())
        self.bias = bias
        self.nonlinear = nonlinear
        self.p = p
        self.bn = bn

    
    def forward(self, x):
        x = reduce(lambda x,layer: forward_layer(layer, x), self.layers,x)
        return x
    
    def top_net(self, top):
        # start from a clone
        top_net = copy.deepcopy(self)
        
        # general index
        top_net.layers=top_net.layers[top:]
        
        # how many conv layers to throw
        if top_net.conv_layers is not None:
            top_net.conv_layers = top_net.conv_layers[top:]
            if len(top_net.conv_layers) == 0:
                top_net.conv_layers=None
                top = top - len(self.conv_layers)
            else:
                top = top - (len(self.conv_layers) - len(top_net.conv_layers))
            
        # throw fc_layers as needed
        top_net.fc_layers = top_net.fc_layers[top:]
        
        return top_net
        
    def fc_layer_activations(self, layer_idx, x, layer_only=False):
        starting_layer = layer_idx if layer_only else 0
        x = reduce(lambda x, f:f(x), self.fc_layers[starting_layer:layer_idx+1], x)
        return x
    
    def layer_activations(self, layer_idx, x, layer_only=False):
        starting_layer = layer_idx if layer_only else 0
        x = reduce(lambda x,layer: forward_layer(layer, x), self.layers[starting_layer:layer_idx+1],x)
        return x


def layer_activations(net, inputs, layer_idx=0):
    return net.layer_activations(layer_idx, inputs)


def get_bn_labels_preds(dataset, net, net_outputs_fn=None, device=None):
    inputs, labels = dataset
    inputs, labels = (Variable(inputs).cuda(), Variable(labels).cuda()) if device is None else (Variable(inputs).to(device), Variable(labels).to(device))
#     inputs, labels = Variable(inputs), Variable(labels)
    if net_outputs_fn is None:
        outputs = net(inputs)
    else:
        outputs = net_outputs_fn(net, inputs)
    return labels, outputs

def get_inputs_labels(local_batch, device):
    inputs, labels = local_batch
    return Variable(inputs).to(device), Variable(labels).to(device)

def get_loss(criterion, labels, outputs, loss_factor=1.):
    loss = criterion(outputs, labels) * loss_factor
    return loss

def accuracy(labels, outputs):
    if outputs.ndimension() > 1:
        predicted = torch.argmax(outputs.data, dim=1)
    total = labels.size(0)
    if labels.ndimension() > 1:
        labels = torch.argmax(labels, dim=1)
    correct = (predicted == labels).sum().item()
    return correct/total*100

def get_layer_name(net, layer_idx):
    return list(net.layers.named_children())[layer_idx]

def get_layer_name_x0(net, layer_idx):    
    name,layer=get_layer_name(net, layer_idx)
    w0=layer.module.weight.detach().numpy().reshape(-1)#.reshape(layer.module.weight.shape)
    if layer.module.bias is not None:
        b0=layer.module.bias.detach().numpy()
        x0=np.concatenate([w0,b0], axis=0)
    else:
        x0=w0
    return name, x0

def get_layer_idx_name_x0(net, layer_idx):    
    name,layer=get_layer_name(net, layer_idx)
    w0=(
        layer.module
        .weight
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)#.reshape(layer.module.weight.shape)
    )
    if layer.module.bias is not None:
        b0=layer.module.bias.detach().cpu().numpy()
        x0=np.concatenate([w0,b0], axis=0)
    else:
        x0=w0
    x0=copy.deepcopy(x0)
    return layer_idx,name,x0

def update_layer_(x, net, layer_idx, device=torch.device('cpu'), permute_layer=False):
    x=copy.deepcopy(x)
    name,layer=get_layer_name(net, layer_idx)
    module = layer.module
    weight_shape=module.weight.shape
    # update weights and bias
    weight_bias_split_idx=reduce(lambda a,b:a*b, weight_shape)
    weight = x[:weight_bias_split_idx]
    if permute_layer:
        weight = np.random.permutation(weight)
    module.weight.data = torch.tensor(
        weight,
        dtype=module.weight.dtype,
        device=device,
    ).reshape(weight_shape)
    if module.bias is not None:
        bias=x[weight_bias_split_idx:]
        if permute_layer:
            bias = np.random.permutation(bias)
        module.bias.data = torch.tensor(
            bias,
            dtype=module.bias.dtype,
            device=device,
        )