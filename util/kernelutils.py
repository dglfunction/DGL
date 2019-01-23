from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from itertools import cycle

import re

from functools import reduce

import scipy
from itertools import product


import time

from util.exp import (
    weight_init
)

from util.exp import (
    FFNet,ActivatedLinear,)#FFNetDropout)

from .util import *

eps = 1e-6

# ***********
# ReLU Kernel 
# ***********

def oneLayer_np(kxx,kxxt,kxtxt,w_sig,b_sig): # implements a single mean-field layer as in the GP paper of Yasaman for ReLU
    theta_xxt = np.arccos((kxxt/(kxx*kxtxt+eps)**(0.5))*(1-eps))
    
    ret_kxxt = b_sig + (w_sig/(2*np.pi))*(kxx*kxtxt+eps)**(0.5)*(np.sin(theta_xxt)+(np.pi-theta_xxt)*np.cos(theta_xxt))
    ret_kxx =  b_sig + (w_sig/(2*np.pi))*kxx*((np.pi))
    ret_kxtxt =  b_sig + (w_sig/(2*np.pi))*kxtxt*((np.pi))
    
    
        
    
    return (ret_kxxt,ret_kxx,ret_kxtxt)

def ReLUKernel_np(x,xt,wb_sig_list): # The last index of x and xt would be acted upon with K
    L = len(wb_sig_list)//2
    (w0,b0) = (wb_sig_list[0],wb_sig_list[1])
    kxxt = np.sum(x*xt,axis=len(x.shape)-1)*w0 + b0 
    kxx = np.sum(x*x,axis=len(x.shape)-1)*w0 + b0
    kxtxt = np.sum(xt*xt,axis=len(x.shape)-1)*w0 + b0  
    for l in range(1,L):
        (w0,b0) = (wb_sig_list[2*l],wb_sig_list[2*l+1])
        (kxxt,kxx,kxtxt) = oneLayer_np(kxx,kxxt,kxtxt,w0,b0)
   
    return kxxt
from IPython.core.debugger import set_trace

def oneLayer(kxx,kxxt,kxtxt,w_sig,b_sig,eps): # implements a single mean-field layer as in the GP paper of Yasaman for ReLU
    
    theta_xxt = torch.acos((kxxt/torch.sqrt(kxx*kxtxt+eps))*(1-eps))
    
    ret_kxxt = torch.sqrt(kxx*kxtxt+eps)*(torch.sin(theta_xxt)+(np.pi-theta_xxt)*torch.cos(theta_xxt))*(w_sig/(2*np.pi))+b_sig
    ret_kxx = kxx*((np.pi))*(w_sig/(2*np.pi)) + b_sig
    ret_kxtxt = kxtxt*((np.pi))*(w_sig/(2*np.pi)) + b_sig
    
  #  if (torch.min(kxx*kxtxt) < 100*eps) or (torch.min(kxtxt*kxtxt) < 100*eps) or (torch.min(kxx*kxx) < 100*eps):
  #      print(('possible eps choice issue: ',torch.min(kxx*kxtxt),torch.min(kxx*kxx),torch.min(kxtxt*kxtxt)))
        
    
    return (ret_kxxt,ret_kxx,ret_kxtxt)
   #return (kxxt,kxx,kxtxt)


def ReLUKernel(x,xt,wb_sig_list,eps=eps): # The last index of x and xt would be acted upon with K. Rest are broadcasted
    L = len(wb_sig_list)//2
    (w0,b0) = (wb_sig_list[0],wb_sig_list[1])
    kxxt = (x*xt).sum(dim=len(x.shape)-1)*w0 + b0
    kxx = (x*x).sum(dim=len(x.shape)-1)*w0 + b0
    kxtxt =(xt*xt).sum(dim=len(x.shape)-1)*w0 + b0
    
    for l in range(1,L):
        (w0,b0) = (wb_sig_list[2*l],wb_sig_list[2*l+1])
        (kxxt,kxx,kxtxt) = oneLayer(kxx,kxxt,kxtxt,w0,b0,eps=eps)
   
    return kxxt

# **********
# Erf Kernel
# **********

def oneLayer_erf_np(kxx,kxxt,kxtxt,w_sig,b_sig): # implements a single mean-field layer as in the GP paper of Yasaman for ReLU
    
    fac_xx = np.sqrt(1+2*kxx)
    fac_xtxt = np.sqrt(1+2*kxtxt)
    
    ret_kxxt = b_sig + (2*w_sig/(np.pi))*np.arcsin(2*kxxt/(fac_xx*fac_xtxt))
    ret_kxx =  b_sig + (2*w_sig/(np.pi))*np.arcsin(2*kxx/(fac_xx*fac_xx))
    ret_kxtxt =  b_sig + (2*w_sig/(np.pi))*np.arcsin(2*kxtxt/(fac_xtxt*fac_xtxt))
    
    return (ret_kxxt,ret_kxx,ret_kxtxt)

def ErfKernel_np(x,xt,wb_sig_list): # The last index of x and xt would be acted upon with K
    L = len(wb_sig_list)//2
    (w0,b0) = (wb_sig_list[0],wb_sig_list[1])
    kxxt = np.sum(x*xt,axis=len(x.shape)-1)*w0 + b0 
    kxx = np.sum(x*x,axis=len(x.shape)-1)*w0 + b0
    kxtxt = np.sum(xt*xt,axis=len(x.shape)-1)*w0 + b0  
    for l in range(1,L):
        (w0,b0) = (wb_sig_list[2*l],wb_sig_list[2*l+1])
        (kxxt,kxx,kxtxt) = oneLayer_erf_np(kxx,kxxt,kxtxt,w0,b0)
   
    return kxxt
from IPython.core.debugger import set_trace

def oneLayer_erf(kxx,kxxt,kxtxt,w_sig,b_sig,eps): # implements a single mean-field layer as in the GP paper of Yasaman for ReLU
   
    fac_xx = torch.sqrt(1+2*kxx)
    fac_xtxt = torch.sqrt(1+2*kxtxt)
    
    ret_kxxt = b_sig + (2*w_sig/(np.pi))*torch.arcsin(2*kxxt/(fac_xx*fac_xtxt))
    ret_kxx =  b_sig + (2*w_sig/(np.pi))*torch.arcsin(2*kxx/(fac_xx*fac_xx))
    ret_kxtxt =  b_sig + (2*w_sig/(np.pi))*torch.arcsin(2*kxtxt/(fac_xtxt*fac_xtxt))
    
    return (ret_kxxt,ret_kxx,ret_kxtxt)


def ErfKernel(x,xt,wb_sig_list,eps=eps): # The last index of x and xt would be acted upon with K. Rest are broadcasted
    L = len(wb_sig_list)//2
    (w0,b0) = (wb_sig_list[0],wb_sig_list[1])
    kxxt = (x*xt).sum(dim=len(x.shape)-1)*w0 + b0
    kxx = (x*x).sum(dim=len(x.shape)-1)*w0 + b0
    kxtxt =(xt*xt).sum(dim=len(x.shape)-1)*w0 + b0
    for l in range(1,L):
        (w0,b0) = (wb_sig_list[2*l],wb_sig_list[2*l+1])
        (kxxt,kxx,kxtxt) = oneLayer_erf(kxx,kxxt,kxtxt,w0,b0,eps=eps)
   
    return kxxt

# *****************************
# Kernel non-specific functions  
# *****************************

def set_norm(net, wnorms2, bnorms2):
    # Note that this assumes exp/weight_init is set to xavier_init with tanh gain for the weights and std=0.5 for the biases. 
    n_activated = len(net.layers)-1
    norms = torch.Tensor(np.sqrt(wnorms2))
    if net.bias:
        bnorms = torch.Tensor(np.sqrt(bnorms2))
    #norms = norms[5-n_h_layers:-1]
    
    # activated layers
    for i in range(n_activated):
        inv_xavier_tanh_factor = (3./5.)*(
            (2./(net.layers[i].module.weight.shape[0]+net.layers[i].module.weight.shape[1]))**(-0.5)
        )
        
        w_num = net.layers[i].module.weight.shape[0]*net.layers[i].module.weight.shape[1]
        net.layers[i].module.weight.data = net.layers[i].module.weight.mul(inv_xavier_tanh_factor*norms[i]/(w_num**0.5))
        
        if net.bias:
            b_num = net.layers[i].module.bias.shape[0]
            net.layers[i].module.bias.data = net.layers[i].module.bias.mul(bnorms[i]*2./(b_num**0.5))
    
    # linear classifier layer
    i = n_activated 
    
    inv_xavier_tanh_factor = (3./5.)*(
        (2./(net.layers[i].weight.shape[0]+net.layers[i].weight.shape[1]))**(-0.5)
    )
    
    w_num = net.layers[i].weight.shape[0]*net.layers[i].weight.shape[1]
    net.layers[i].weight.data = net.layers[i].weight.mul(inv_xavier_tanh_factor*norms[i]/(w_num**0.5))
    
    if net.bias:
        b_num = net.layers[i].bias.shape[0]
        net.layers[i].bias.data = net.layers[i].bias.mul(bnorms[i]*2./(b_num**0.5))
        
def sample_network_corr(net,n_input,wnorms2,bnorms2,stat,N=200,n_outputs=2,skip_train=False):
    # Generate kernel samples dataset and testset  
    
   
    lim = 10.
    res = lim/N

   
    pic = np.zeros((N,N))

    vec1 = np.zeros((n_input,))
    vec2 = np.zeros((n_input,))
    vec1[0] = 0.7
    vec1[1] = 0.
    vec2[0] = 0.
    vec2[1] = 0.7

    vec1b = np.zeros((n_input,))
    vec2b = np.zeros((n_input,))
    vec1b[0] = 2.
    vec1b[1] = 2.
    vec2b[0] = -2.
    vec2b[1] = 2.

    vec1_test = np.zeros((n_input,))
    vec2_test = np.zeros((n_input,))
    vec1_test[1] = 1.
    vec1_test[2] = 2.
    vec2_test[1] = -2.
    vec2_test[2] = 1.


    N_slice = 1

    preds12 = np.zeros((stat*4,N,N))
    preds12_test = np.zeros((stat,N,N))


    for n in range(stat*4 if not skip_train else 2):
            net.apply(weight_init)
            set_norm(net,wnorms2,bnorms2)
            vecs1 = [[(x/N)*vec1] for x in range(N//2)]+[[(x/N)*vec1b] for x in range(N//2)]
            vecs2 = [[vec2*np.sin(2*np.pi*x/N)+vec1*np.cos(2*np.pi*x/N)] for x in range(N//2)]+[[vec2b*np.sin(2*np.pi*x/N)+vec1b*np.cos(2*np.pi*x/N)] for x in range(N//2)]
            
            vecs1_ten = torch.Tensor(vecs1)
            vecs2_ten = torch.Tensor(vecs2)
          

            preds1 = net(vecs1_ten).detach().numpy()
            preds2 = net(vecs2_ten).detach().numpy()
            preds2 = preds2.reshape((N,n_outputs))
            preds1 = preds1.reshape((N,n_outputs))

            preds12[n] = (preds1[np.newaxis,:,:]*preds2[:,np.newaxis,:])[:,:,0]
            
    for n in range(stat):
            net.apply(weight_init)
            set_norm(net,wnorms2,bnorms2)
           

            vecs_test = [[np.sin(2*np.pi*x/N)*vec1_test + np.cos(2*np.pi*x/N)*vec2_test] for x in range(N)]
            
            vecs_test = torch.Tensor(vecs_test)

            preds_test = net(vecs_test).detach().numpy()
           
            preds_test = preds_test.reshape((N,n_outputs))
          
            preds12_test[n] = (preds_test[np.newaxis,:,:]*preds_test[:,np.newaxis,:])[:,:,0]
            

    #print(np.max(preds12))

    pic = np.average(preds12,axis=0)
    pic_var = np.var(preds12,axis=0)
    # regularize variance to > eps
    pic_var = np.where(pic_var > eps, pic_var, eps*np.ones_like(pic_var))
    
    pic_test = np.average(preds12_test,axis=0)
    pic_test_var = np.var(preds12_test,axis=0)
    # regularize variance to > eps
    pic_test_var = np.where(pic_test_var > eps, pic_test_var, eps*np.ones_like(pic_test_var))
                       
    vecs1 = np.array(vecs1)
    vecs2 = np.array(vecs2)
    vecs_test = np.array(vecs_test)
    
    return vecs1,vecs2,vecs_test,pic,pic_var,pic_test,pic_test_var 

def get_kernel_params(net,act_type='ReLU'): # Estimates the kernel of the top network of net 
    # Gather data from net 
    n_activated = len(net.layers)-1
    n_outputs = net.layers[-1].weight.shape[0]
   

    if (n_activated > 0): 
        n_input = net.layers[0].module.weight.shape[1]
    else:
        n_input = net.layers[0].weight.shape[1]
    
    
    wnorms2 = [net.layers[i].module.weight.pow(2).sum().detach().numpy() for i in range(n_activated)]
    wnorms2 = wnorms2 + [net.layers[n_activated].weight.pow(2).sum().detach().numpy()]
    if net.bias:
        bnorms2 = [net.layers[i].module.bias.pow(2).sum().detach().numpy() for i in range(n_activated)]
        bnorms2 = bnorms2 + [net.layers[n_activated].bias.pow(2).sum().detach().numpy()]
    else:
        bnorms2 = [0]*(n_activated+1)
        
    # Get W_{ij} and b_j distribution (normalization of previous by size) 
    # This is a bit tricky : All layer except the input layer have K(x,x') = b_j^2 + d_{in} w_{ij}^2 E[previous]
    # So there is some typo or implicit normalization issue in the Yasaman's paper. 
    # However the input layer specifically is K^0(x,x') = b_j^2 + w_{ij}^2 x \cdot x' 
    # Hence everything is normalized by shape[0], since w.pow(2).sum() = shape[0]*d_{in}*w_{ij}, except the input layer.
    wnorms2_i = [(net.layers[i].module.weight.pow(2).sum()/(net.layers[i].module.weight.shape[0])).detach().numpy() for i in range(n_activated)]
    wnorms2_i = wnorms2_i + [(net.layers[n_activated].weight.pow(2).sum()/(net.layers[n_activated].weight.shape[0])).detach().numpy()]
    if net.bias: 
        bnorms2_i = [(net.layers[i].module.bias.pow(2).sum()/net.layers[i].module.bias.shape[0]).detach().numpy() for i in range(n_activated)]
        bnorms2_i = bnorms2_i + [(net.layers[n_activated].bias.pow(2).sum()/net.layers[n_activated].bias.shape[0]).detach().numpy()]
    else:
        bnorms2_i = [0]*(n_activated+1)
    
    wnorms2_i[0] = wnorms2_i[0]/n_input
    
  #  print("norm W_i: ", np.sqrt(wnorms2_i))
  #  print("norm b_i: ", np.sqrt(bnorms2_i))
    
    # Initialized optimization to mean-field starting point 
    wb_sig_list = [[],[],[],[]]
    for i in range(n_activated+1):
        wb_sig_list[0] += [np.sqrt(wnorms2_i[i])]
        wb_sig_list[0] += [np.sqrt(bnorms2_i[i])]
        wb_sig_list[1] += [3*np.sqrt(wnorms2_i[i])]  
        wb_sig_list[1] += [3*np.sqrt(bnorms2_i[i])]
        wb_sig_list[2] += [0.3*np.sqrt(wnorms2_i[i])]
        wb_sig_list[2] += [0.3*np.sqrt(bnorms2_i[i])]
        wb_sig_list[3] += [0.1]
        wb_sig_list[3] += [0.1]
       
   
    
   # print("norm W: ", np.sqrt(wnorms2))
   # print("norm b: ", np.sqrt(bnorms2))
    # Generate kernel samples dataset 
    stat = 10000
    N = 200
    vecs1,vecs2,vecs_test,pic,pic_var,pic_test,pic_test_var = sample_network_corr(net,n_input,wnorms2,bnorms2,stat,N,n_outputs)
    
    # Fit to mean-field ansatz, output potential failure message 
    
    if act_type == 'ReLU':
        f1 = lambda params: ReLUKernel_np(vecs1[np.newaxis,:,:],vecs2[:,np.newaxis,:],np.array(params)**2).reshape(N,N)
    elif act_type == 'erf':
        f1 = lambda params: ErfKernel_np(vecs1[np.newaxis,:,:],vecs2[:,np.newaxis,:],np.array(params)**2).reshape(N,N)
    else:
        print('Unsupported activation')
    l2 = lambda params: (f1(params)-pic).reshape((N**2))
    
    costs = []
    results = [] 
    for n in range(4):
        optRes = scipy.optimize.leastsq(l2, wb_sig_list[n],full_output=True,xtol=0.00001) 
        results += [optRes[0]]
        costs += [np.sum((optRes[2]['fvec'])**2)]
    print("costs : ",costs)
    res_x = results[np.argmin(costs)]
    
    # test accuracy on the testset 
     
    if act_type == 'ReLU':
        f1_test = lambda params: ReLUKernel_np(vecs_test[np.newaxis,:,:],vecs_test[:,np.newaxis,:],np.array(params)**2).reshape(N,N)
    elif act_type == 'erf':
        f1_test = lambda params: ErfKernel_np(vecs_test[np.newaxis,:,:],vecs_test[:,np.newaxis,:],np.array(params)**2).reshape(N,N)
    else:
        print('Unsupported activation')
    l2_test = lambda params: (f1_test(params)-pic_test).reshape((N**2))
    
    acc = np.sqrt(np.sum(l2_test(res_x)**2/((0*pic_test**2 + pic_test_var/stat).reshape(N**2)))/((N**2)))
    #print("test scaled L2:",acc)
    
    #print("parameter list ",res_x)
    return res_x,wnorms2,bnorms2,acc #pic,l2(res_x).reshape(N,N)+pic



def get_MF_params(net,act_type='ReLU'): # Estimates the kernel of the top network of net 
    # Gather data from net 
    n_activated = len(net.layers)-1
    if (n_activated > 0): 
        n_input = net.layers[0].module.weight.shape[1]
    else:
        n_input = net.layers[0].weight.shape[1]
    
    
    wnorms2 = [net.layers[i].module.weight.pow(2).sum().item() for i in range(n_activated)]
    wnorms2 = wnorms2 + [net.layers[n_activated].weight.pow(2).sum().item()]
    if net.bias:
        bnorms2 = [net.layers[i].module.bias.pow(2).sum().item() for i in range(n_activated)]
        bnorms2 = bnorms2 + [net.layers[n_activated].bias.pow(2).sum().item()]
    else:
        bnorms2 = [0]*(n_activated+1)
        
    # Get W_{ij} and b_j distribution (normalization of previous by size) 
    # This is a bit tricky : All layer except the input layer have K(x,x') = b_j^2 + d_{in} w_{ij}^2 E[previous]
    # So there is some typo or implicit normalization issue in the Yasaman's paper. 
    # However the input layer specifically is K^0(x,x') = b_j^2 + w_{ij}^2 x \cdot x' 
    # Hence everything is normalized by shape[0], since w.pow(2).sum() = shape[0]*d_{in}*w_{ij}, except the input layer.
    wnorms2_i = [(net.layers[i].module.weight.pow(2).sum()/(net.layers[i].module.weight.shape[0])).item() for i in range(n_activated)]
    wnorms2_i = wnorms2_i + [(net.layers[n_activated].weight.pow(2).sum()/(net.layers[n_activated].weight.shape[0])).item()]
    if net.bias: 
        bnorms2_i = [(net.layers[i].module.bias.pow(2).sum()/net.layers[i].module.bias.shape[0]).item() for i in range(n_activated)]
        bnorms2_i = bnorms2_i + [(net.layers[n_activated].bias.pow(2).sum()/net.layers[n_activated].bias.shape[0]).item()]
    else:
        bnorms2_i = [0]*(n_activated+1)
    
    wnorms2_i[0] = wnorms2_i[0]/n_input
    
  #  print("norm W_i: ", np.sqrt(wnorms2_i))
  #  print("norm b_i: ", np.sqrt(bnorms2_i))

    wb_sig_list = []
    for i in range(n_activated+1):
        wb_sig_list += [np.sqrt(wnorms2_i[i])]
        wb_sig_list += [np.sqrt(bnorms2_i[i])]
   
    return wb_sig_list



# Dropout compatible versions 
# ***************************

def set_norm_drop(net,wnorms2,bnorms2):
        n_activated = len(net.fc_layers)-1
        norms = torch.Tensor(np.sqrt(wnorms2))
        bnorms = torch.Tensor(np.sqrt(bnorms2))
#norms = norms[5-n_h_layers:-1]
        i = 0 
        for layer in net.fc_layers:
            if type(layer) == ActivatedLinear:
                layer.fc.weight.data = layer.fc.weight.mul(norms[i])/layer.fc.weight.norm()
                layer.fc.bias.data = layer.fc.bias.mul(bnorms[i])/layer.fc.bias.norm()
                i = i+1
            if type(layer) == torch.nn.modules.linear.Linear:
                layer.weight.data = layer.weight.data*norms[i]/layer.weight.norm()
                layer.bias.data = layer.bias.data*bnorms[i]/layer.bias.norm()

def get_relu_kernel_params_drop(net): # Estimates the kernel of the top network of net 
    # Gather data from net 
    if (len(net.fc_layers)>1): 
        n_input = net.fc_layers[0].fc.weight.shape[1]
    else:
        n_input = net.fc_layers[0].weight.shape[1]
   
    wnorms2 = []
    bnorms2 = []
    for layer in net.fc_layers:
        if type(layer) == ActivatedLinear:
            wnorms2 += [layer.fc.weight.pow(2).sum().detach().numpy()]
            bnorms2 += [layer.fc.bias.pow(2).sum().detach().numpy()]
            
        if type(layer) == torch.nn.modules.linear.Linear:
            wnorms2 += [layer.weight.pow(2).sum().detach().numpy()]
            bnorms2 += [layer.bias.pow(2).sum().detach().numpy()]
    
    wb_sig_list = [1.]*(len(wnorms2)*2)
       
    print("norm W: ", np.sqrt(wnorms2))
    print("norm b: ", np.sqrt(bnorms2))
    # Generate kernel samples dataset 
    
   # wb_sig_list = []
   # for i in range(n_activated+1):
   #     wb_sig_list = wb_sig_list + [np.sqrt(wnorms2[i].detach().numpy()/100.)] + [np.sqrt(bnorms2[i].detach().numpy()/100.)]
   # wb_sig_list += [1.]
    
    N = 200
    lim = 10.
    res = lim/N

    stat = 16000
    pic = np.zeros((N,N))

    vec1 = np.zeros((n_input,))
    vec2 = np.zeros((n_input,))
    vec1[0] = 0.7
    vec1[1] = 0.
    vec2[0] = 0.
    vec2[1] = 0.7

    vec1b = np.zeros((n_input,))
    vec2b = np.zeros((n_input,))
    vec1b[0] = 1.
    vec1b[1] = 1.
    vec2b[0] = -1.
    vec2b[1] = 1.



    N_slice = 1

    preds12 = np.zeros((stat,N,N))

    net.train()
    for n in range(stat):
            net.apply(weight_init)
            set_norm_drop(net,wnorms2,bnorms2)
            vecs1 = [[(x/N)*vec1] for x in range(N//2)]+[[(x/N)*vec1b] for x in range(N//2)]
            vecs2 = [[vec2*np.sin(2*np.pi*x/N)+vec1*np.cos(2*np.pi*x/N)] for x in range(N//2)]+[[vec2b*np.sin(2*np.pi*x/N)+vec1b*np.cos(2*np.pi*x/N)] for x in range(N//2)]


            vecs1_ten = torch.Tensor(vecs1)
            vecs2_ten = torch.Tensor(vecs2)

            preds1 = net(vecs1_ten).detach().numpy()
            preds2 = net(vecs2_ten).detach().numpy()
            preds2 = preds2.reshape((N,2))
            preds1 = preds1.reshape((N,2))

            preds12[n] = np.sum(preds1[np.newaxis,:,:]*preds2[:,np.newaxis,:],axis=2)

    #print(np.max(preds12))

    pic = np.average(preds12,axis=0)

    vecs1 = np.array(vecs1)
    vecs2 = np.array(vecs2)
    
    # Fit to mean-field ansatz, output potential failure message 
    
    f1 = lambda params: ReLUKernel_np(vecs1[np.newaxis,:,:],vecs2[:,np.newaxis,:],np.array(params)**2).reshape(N,N)
    l2 = lambda params: (f1(params)-pic).reshape((N**2))
    
    optRes = scipy.optimize.leastsq(l2, wb_sig_list,full_output=True,xtol=0.00001) #,ftol=(N**2)*0.01**2) # We allow for 1 percent error on the training set. From a few experiments I've done, error on unseen pairs of point is likely to be 2 or 3 times as large.
    
    res_x = optRes[0]
  
    print("(L2Scaled/Excepted-L2Scaled-with-1%-train-error)^(0.5) (Average mistake in units of 1%): ",np.sqrt(np.sum(l2(res_x)**2/(0.5+pic.reshape(N**2)**2))/((N**2)*(0.01**2))))
   
    return res_x,wnorms2,bnorms2 #pic,l2(res_x).reshape(N,N)+pic


def top_network_drop(net,top):
    n_activated = 0
    l = net.fc_layers[0]
    if (type(l)==torch.nn.modules.linear.Linear):     
        n_input = l.weight.shape[1]
        h_h = l.weight.shape[0]
    else:
        n_input = l.fc.weight.shape[1]
        h_h = l.fc.weight.shape[0] 
        
    for l in net.fc_layers:
        if type(l)==ActivatedLinear:
            n_activated += 1
        
           
    top_net = FFNetDropout(
    n_inputs=n_input, 
    n_classes=2, 
    n_h_layers=n_activated-1-top, 
    h_h_layers=h_h,
    nonlinear=F.relu,
    bn=False,
    )
    
    cur_idx = 0
    for l in top_net.fc_layers:
        if type(l) == ActivatedLinear:
            l.fc.weight.data = net.fc_layers[cur_idx+2*top].fc.weight.data.clone()
            l.fc.bias.data = net.fc_layers[cur_idx+2*top].fc.bias.data.clone()
            cur_idx += 2
        if type(l) == torch.nn.modules.linear.Linear:
            l.weight.data = net.fc_layers[-1].weight.data.clone()
            l.bias.data = net.fc_layers[-1].bias.data.clone()
    
    return top_net



# ****************
# Mean field tests
# ****************

def test_MF_params(net,act_type='ReLU'): # Estimates the kernel of the top network of net  
    # Gather data from net 
  
    n_activated = len(net.layers)-1
    n_outputs = net.layers[-1].weight.shape[0]
    
    if (n_activated > 0): 
        n_input = net.layers[0].module.weight.shape[1]
    else:
        n_input = net.layers[0].weight.shape[1]
       
    
    wnorms2 = [net.layers[i].module.weight.pow(2).sum().detach().numpy() for i in range(n_activated)]
    wnorms2 = wnorms2 + [net.layers[n_activated].weight.pow(2).sum().detach().numpy()]
    if net.bias:
        bnorms2 = [net.layers[i].module.bias.pow(2).sum().detach().numpy() for i in range(n_activated)]
        bnorms2 = bnorms2 + [net.layers[n_activated].bias.pow(2).sum().detach().numpy()]
    else:
        bnorms2 = [0]*(n_activated+1)
    
    # Get W_{ij} and b_j distribution (normalization of previous by size) 
    # This is a bit tricky : All layer except the input layer have K(x,x') = b_j^2 + d_{in} w_{ij}^2 E[previous]
    # So there is some typo or implicit normalization issue in the Yasaman's paper. 
    # However the input layer specifically is K^0(x,x') = b_j^2 + w_{ij}^2 x \cdot x' 
    # Hence everything is normalized by shape[0], since w.pow(2).sum() = shape[0]*d_{in}*w_{ij}, except the input layer.
    wnorms2_i = [(net.layers[i].module.weight.pow(2).sum()/(net.layers[i].module.weight.shape[0])).detach().numpy() for i in range(n_activated)]
    wnorms2_i = wnorms2_i + [(net.layers[n_activated].weight.pow(2).sum()/(net.layers[n_activated].weight.shape[0])).detach().numpy()]
    if net.bias:
        bnorms2_i = [(net.layers[i].module.bias.pow(2).sum()/net.layers[i].module.bias.shape[0]).detach().numpy() for i in range(n_activated)]
        bnorms2_i = bnorms2_i + [(net.layers[n_activated].bias.pow(2).sum()/net.layers[n_activated].bias.shape[0]).detach().numpy()]
    else:
        bnorms2_i = [0]*(n_activated+1)
    wnorms2_i[0] = wnorms2_i[0]/n_input
    
  #  print("norm W_i: ", np.sqrt(wnorms2_i))
  #  print("norm b_i: ", np.sqrt(bnorms2_i))
    
    # pack it into a list 
    res_x = []
    for i in range(n_activated+1):
        res_x += [np.sqrt(wnorms2_i[i])]
        res_x += [np.sqrt(bnorms2_i[i])]
    
    
    
    # Generate kernel samples dataset 
    
    N  = 200
    stat = 10000
    vecs1,vecs2,vecs_test,pic,pic_var,pic_test,pic_test_var = sample_network_corr(net,n_input,wnorms2,bnorms2,stat,N,n_outputs=2,skip_train=True)
  
    
    
    
    # Compare to mean-field  
    
    if act_type == 'ReLU':
        f1 = lambda params: ReLUKernel_np(vecs_test[np.newaxis,:,:],vecs_test[:,np.newaxis,:],np.array(params)**2).reshape(N,N)
    elif act_type == 'erf':
        f1 = lambda params: ErfKernel_np(vecs_test[np.newaxis,:,:],vecs_test[:,np.newaxis,:],np.array(params)**2).reshape(N,N)
    else:
        print('Unsupported activation')
    l2 = lambda params: (f1(params)-pic_test).reshape((N**2))
    
    
    acc = np.sqrt(np.sum(l2(res_x)**2/((0*pic_test**2+pic_test_var/stat).reshape(N**2)))/(N**2))
    
     
    return acc #pic,l2(res_x).reshape(N,N)+pic



def test_MF_params_on_data(net,input_data,act_type='ReLU'): # Estimates the kernel of the top network of net  
    # Gather data from net 
  
    n_activated = len(net.layers)-1
  
    if (n_activated > 0): 
        n_input = net.layers[0].module.weight.shape[1]
    else:
        n_input = net.layers[0].weight.shape[1]
       
    n_outputs = net.layers[-1].weight.shape[0]
        
    wnorms2 = [net.layers[i].module.weight.pow(2).sum().detach().numpy() for i in range(n_activated)]
    wnorms2 = wnorms2 + [net.layers[n_activated].weight.pow(2).sum().detach().numpy()]
    if net.bias:
        bnorms2 = [net.layers[i].module.bias.pow(2).sum().detach().numpy() for i in range(n_activated)]
        bnorms2 = bnorms2 + [net.layers[n_activated].bias.pow(2).sum().detach().numpy()]
    else:
        bnorms2 = [0]*(n_activated+1)
    
    # Get W_{ij} and b_j distribution (normalization of previous by size) 
    # This is a bit tricky : All layer except the input layer have K(x,x') = b_j^2 + d_{in} w_{ij}^2 E[previous]
    # So there is some typo or implicit normalization issue in the Yasaman's paper. 
    # However the input layer specifically is K^0(x,x') = b_j^2 + w_{ij}^2 x \cdot x' 
    # Hence everything is normalized by shape[0], since w.pow(2).sum() = shape[0]*d_{in}*w_{ij}, except the input layer.
    wnorms2_i = [(net.layers[i].module.weight.pow(2).sum()/(net.layers[i].module.weight.shape[0])).detach().numpy() for i in range(n_activated)]
    wnorms2_i = wnorms2_i + [(net.layers[n_activated].weight.pow(2).sum()/(net.layers[n_activated].weight.shape[0])).detach().numpy()]
    if net.bias:
        bnorms2_i = [(net.layers[i].module.bias.pow(2).sum()/net.layers[i].module.bias.shape[0]).detach().numpy() for i in range(n_activated)]
        bnorms2_i = bnorms2_i + [(net.layers[n_activated].bias.pow(2).sum()/net.layers[n_activated].bias.shape[0]).detach().numpy()]
    else:
        bnorms2_i = [0]*(n_activated+1)
    wnorms2_i[0] = wnorms2_i[0]/n_input
    
  #  print("norm W_i: ", np.sqrt(wnorms2_i))
  #  print("norm b_i: ", np.sqrt(bnorms2_i))
    
    # pack it into a list 
    res_x = []
    for i in range(n_activated+1):
        res_x += [np.sqrt(wnorms2_i[i])]
        res_x += [np.sqrt(bnorms2_i[i])]
    
    
    
    # Generate kernel samples dataset 
    
    N  = input_data.shape[0]
    stat = 10000
    # Generate kernel samples dataset and testset  
    
    preds12_test = np.zeros((stat,N,N))

    for n in range(stat):
            net.apply(weight_init)
            set_norm(net,wnorms2,bnorms2)
           

            vecs_test = input_data
            
            preds_test = net(vecs_test).detach().numpy()
           
            preds_test = preds_test.reshape((N,n_outputs))
          
            preds12_test[n] = (preds_test[np.newaxis,:,:]*preds_test[:,np.newaxis,:])[:,:,0]
            

    #print(np.max(preds12))

    pic_test = np.average(preds12_test,axis=0)
    pic_test_var = np.var(preds12_test,axis=0)
    # regularize variance to > eps
    pic_test_var = np.where(pic_test_var > eps, pic_test_var, eps*np.ones_like(pic_test_var))
                       
    vecs_test = np.array(vecs_test)
   
    # Compare to mean-field  
    
    if act_type == 'ReLU':
        f1 = lambda params: ReLUKernel_np(vecs_test[np.newaxis,:,:],vecs_test[:,np.newaxis,:],np.array(params)**2).reshape(N,N)
    elif act_type == 'erf':
        f1 = lambda params: ErfKernel_np(vecs_test[np.newaxis,:,:],vecs_test[:,np.newaxis,:],np.array(params)**2).reshape(N,N)
    else:
        print('Unsupported activation')
    l2 = lambda params: (f1(params)-pic_test).reshape((N**2))
    
    
    acc = np.sqrt(np.sum(l2(res_x)**2/((0*pic_test**2+pic_test_var/stat).reshape(N**2)))/(N**2))
    
    print("scaled L2:",acc) 
   
    return acc #pic,l2(res_x).reshape(N,N)+pic

# ******************
# BIE LOSS functions
# ******************

def BIE_MSE_loss(inputs_float, labels,K_params,device, dtype=torch.float32,act_type="ReLU",eps=0.2,addSigma=False): # labels are assumed to be samples X one-hot 
    # NOTE- if comparing to pytorch MSE loss - pytorch is also dividing by the labels' vectors dimnesion.
    inputs = inputs_float.type(dtype).to(device)
    labels_f = labels.type(dtype).to(device)

    mb_size = inputs.shape[0]
    
    activations_size = inputs.shape[1]
    
    # Preparing all relevat matrices 
    
    if act_type == "ReLU":
        K = ReLUKernel(inputs[np.newaxis,:,:],inputs[:,np.newaxis,:],K_params)#,net_near_0,net0)
    else:
        K = ErfKernel(inputs[np.newaxis,:,:],inputs[:,np.newaxis,:],K_params)
    U = torch.potrf(K+torch.eye(mb_size).type(dtype).to(device)*eps,upper=True) # K+eps = U^T U 
    U_inv = torch.inverse(U)
    K_inv = torch.mm(U_inv,U_inv.t())
    K_inv_diag = torch.diag(torch.diag(K_inv))
    K_inv_diag_inv = torch.diag((torch.diag(K_inv))**(-1))
    KB = torch.mm(K,K_inv)
    KB_diag = torch.diag(torch.diag(KB))
  
    
    A_left = torch.mm(K_inv,KB_diag*K_inv_diag_inv)
    
    T = KB-torch.eye(mb_size).type(dtype).to(device)-A_left
    
    label_mat = (labels_f[:,np.newaxis,:]*labels_f[np.newaxis,:,:]).sum(dim=2)
    
    loss = torch.trace(torch.mm(torch.mm(T, T.t()),label_mat)) 
    
    # Optional, if you want to make confidence matter to the loss or caputre more faithful < (x-y)^2 > in MSE. Doesn't matter in practice for all the MNIST experiments we did.  
    if addSigma:
        loss += torch.trace(K_inv_diag_inv) - mb_size*eps
    
    
    return loss/mb_size # + 20.*(Nnn-17)**2

def BIE_eps_finder(inputs_float, K_params,device, dtype=torch.float32,act_type="ReLU",acc=0.001): 
    # This EPS finder measures \sqrt{Trace[Sigma^2]} of the dataset in logit space. The idea is that noise which is much small than this spatial scale of the dataset in logit space--- is insignificant. There are several assumptions here: I. \Sigma_{trained} \approx \Sigma_{untrain} II. margin << \Sigma but not margin <<< \Sigma, otherwise \epsilon << \Sigma may smear the margin. 
    inputs = inputs_float.type(dtype).to(device)
    

    mb_size = inputs.shape[0]
    
    activations_size = inputs.shape[1]
    
    # Preparing all relevat matrices 
    
    if act_type == "ReLU":
        K = ReLUKernel(inputs[np.newaxis,:,:],inputs[:,np.newaxis,:],K_params)#,net_near_0,net0)
    else:
        K = ErfKernel(inputs[np.newaxis,:,:],inputs[:,np.newaxis,:],K_params)
    
    
    # return std of K eigenvalues multiplied by acc
    return torch.sqrt(torch.trace(torch.mm(K,K))/mb_size).item()*acc

def BIE_eps_finder_hindsight(inputs_float, K_params,device,act_type="ReLU"): 
   # This EPS finder measures the average std of predictions over the entire dataset. The idea being that adding \epsilon which is much smaller than the uncertaintly already present should make a huge difference. 
   # Issue: For big and easy enough datasets, eps^2 is always much larger than the average \sigma^2(data). For 1/7 binary MNIST 512 seems to be the limit where you get LAPACK errors before they cross.   

    inputs = inputs_float.type(torch.float32)
    

    mb_size = inputs.shape[0]
    
    activations_size = inputs.shape[1]
    
    # Preparing all relevat matrices 
    
    if act_type == "ReLU":
        K = ReLUKernel(inputs[np.newaxis,:,:],inputs[:,np.newaxis,:],K_params)#,net_near_0,net0)
    else:
        K = ErfKernel(inputs[np.newaxis,:,:],inputs[:,np.newaxis,:],K_params)
    eps2 = 1.
    for n in range(40):
        U = torch.potrf(K+torch.eye(mb_size).type(torch.float32).to(device)*eps2,upper=True) # K+eps = U^T U 
        U_inv = torch.inverse(U)
        K_inv = torch.mm(U_inv,U_inv.t())
        K_inv_diag = torch.diag(torch.diag(K_inv))
        K_inv_diag_inv = torch.diag(K_inv)**(-1) - eps2
        sigma2 = torch.sum(K_inv_diag_inv)/mb_size
        print("(sigma2,eps2):",(sigma2,eps2)) 
        if eps2 < 0.3*sigma2:
            break
        else:
            eps2 = eps2/3. 
    return eps2 


    
    
    
    
    
    
        
