import  torch
import  numpy as np
#import torchsnooper
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexFunctions import complex_relu, complex_max_pool2d


class Learner(nn.Module):
    """

    """

    def __init__(self, config, imgc, imgsz):
        """

        :param config: network config file, type:list of (string, list) 元素是元组：名字，参数列表
        :param imgc: 1 or 3
        :param imgsz:  28 or 84 图片大小
        """
        
        #感觉真就只做了一些初始化工作
        super(Learner, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        #self.vars = nn.ParameterList()
        # running_mean and running_var
        #self.vars_bn = nn.ParameterList()#空列表？
        

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                #w = nn.Parameter(torch.ones(*param[:4]))
                #w = nn.Parameter(torch.ones(*param[:4], device='cuda'))
                # gain=1 according to cbfin's implementation
                #torch.nn.init.kaiming_normal_(w)#初始化的结果接在ParameterList后面
                #self.vars.append(w)
                # [ch_out]
                #self.vars.append(nn.Parameter(torch.zeros(param[0], device='cuda')))
                self.conv1=ComplexConv2d(param[1], param[0], (param[2],param[3]), param[4],param[5])

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                #w = nn.Parameter(torch.ones(*param[:4]))
                w = nn.Parameter(torch.ones(*param[:4], device='cuda'))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                #self.vars.append(w)
                # [ch_in, ch_out]
                #self.vars.append(nn.Parameter(torch.zeros(param[1])))
                #self.vars.append(nn.Parameter(torch.zeros(param[1], device='cuda')))

            elif name is 'linear':
                # [ch_out, ch_in]
                #w = nn.Parameter(torch.ones(*param))
                w = nn.Parameter(torch.ones(*param, device='cuda'))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                #self.vars.append(w)
                # [ch_out]
                #self.vars.append(nn.Parameter(torch.zeros(param[0], device='cuda')))
                self.fc1=ComplexLinear(param[1], param[0])

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0], device='cuda'))
                #self.vars.append(w)#也不归一化
                # [ch_out]
                #self.vars.append(nn.Parameter(torch.zeros(param[0], device='cuda')))
                self.bn=ComplexBatchNorm2d(param[0])
                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0], device='cuda'), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0], device='cuda'), requires_grad=False)
                #self.vars_bn.extend([running_mean, running_var])
                
            elif name is 'multihattention':
                multihead_attn=nn.MultiheadAttention(param[0],param[1])
                self.vars_attn=multihead_attn
                self.layer_norm1 = nn.LayerNorm(param[0])
                self.dropout1 = nn.Dropout(p=param[2])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid','dropout','softmax','padding','embedding']:
                #这些层不加额外初始化
                continue
            else:
                raise NotImplementedError






    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn','dropout','softmax','padding','embedding','multihattention']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info


    #@torchsnooper.snoop()
    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """
        #该写的层主要都写forward里了
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        #attn_idx=0
        
        

        for name, param in self.config:
            #print("查看idx")
            #print(name)
            #print(idx)
            if name is 'conv2d':
               # w, b = vars[idx], vars[idx + 1]#id=0 w=vars[0] b=vars[1]
                #print("Atention!!")
                #print(len(w[0]))
                #print(w)
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                print('in:', x.shape)
                x = self.conv1(x)
                #idx += 2
                #print('w的size:',w.shape,'\tout:',b.shape)
                print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                #w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                #print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                #w, b = vars[idx], vars[idx + 1]
                #print("查看idx")
                #print(name)
                #print(idx)
                x = self.fc1(x)
                idx += 2
                #print('forward:', idx, x.norm().item())
                #print(name, param, '\tout:', x.shape)
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = self.bn(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
                #print(name, param, '\tout:', x.shape)
            elif name is 'embedding':
                embeds=nn.Embedding(param[0],param[1])
                x=embeds(torch.LongTensor(x))
                #idx += 2
            elif name is 'multihattention':
                
                x=x.view(-1,64,256)
                x=x.permute(2,0,1)
                #print(x.shape)
                #x=x.type(torch.FloatTensor)
                #x=x.type(torch.cuda.FloatTensor)
                x=x.cuda()
                multihead_attn=self.vars_attn
                x,weights= multihead_attn(x, x, x)
                layer_norm1 = self.layer_norm1
                dropout1 = self.dropout1
                x=layer_norm1(x)
                x=dropout1(x)
                #print(name, param, '\tout:', x.shape)
                x=x.permute(1,2,0)
                x=x.view(-1,64,2,128)
                #print('变形后:', param, '\tout:', x.shape)
                #attn_idx += 1
            elif name is 'flatten':
                #print('flatten前：',x.shape)#无参 全展平了
                x = x.view(x.size(0), -1)
                #print('flatten后：',x.shape)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
                #print(name,  '\tout:', x.shape)
            elif name is 'dropout':
                x=F.dropout(x,p=param[0])
                #print(name,'\tout:', x.shape)
            elif name is 'softmax':
                x=F.softmax(x,dim=0)
                #print(name, '\tout:', x.shape)
            elif name is 'relu':
                x = complex_relu(x, inplace=param[0])
                #print(name, '\tout:', x.shape)
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
                #print(name, param, '\tout:', x.shape)
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
                #print(name, param, '\tout:', x.shape)
            elif name is'padding':
                pad=nn.ZeroPad2d(padding=(param[0],param[0],param[1],param[1]))
                x=pad(x)
                #print(name, param, '\tout:', x.shape)
            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        #在meta中要用到vars这些参数
        #vars=nn.ParameterList()
        #由一个空参数列表 写入参数
        return self.vars