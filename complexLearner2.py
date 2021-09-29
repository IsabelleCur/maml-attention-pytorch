import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
#import torchsnooper
from collaborative_attention import CollaborativeAttention
from complexFunctions import complex_dropout

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
        self.vars = nn.ParameterList().to(torch.device('cuda'))
        #self.varsi = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()#空列表？
        #self.varsi_bn = nn.ParameterList()
        self.vars_attn = nn.ParameterList()
        
        for i, (name, param) in enumerate(self.config):
            if name == 'comp_conv2d':
                #print("!!!---1comp_conv2d---!!!")
                # [ch_out, ch_in, kernelsz, kernelsz]
                #w = nn.Parameter(torch.ones(*param[:4]))
                wr = nn.Parameter(torch.ones(*param[:4]))
                wi = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(wr)#初始化的结果接在ParameterList后面
                torch.nn.init.kaiming_normal_(wi)
                self.vars.append(wr)
                self.vars.append(wi)
                # [ch_out]
                #self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                #for i, p in enumerate(self.vars):
                #    print(i)

            elif name == 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                #w = nn.Parameter(torch.ones(*param[:4]))
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                #self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear':
                # [ch_out, ch_in]
                #w = nn.Parameter(torch.ones(*param))
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                #self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'comp_linear':
                # [ch_out, ch_in]
                #w = nn.Parameter(torch.ones(*param))
                #print("!!!---2comp_linear---!!!")
                wr = nn.Parameter(torch.ones(*param))
                wi = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(wr)
                torch.nn.init.kaiming_normal_(wi)
                self.vars.append(wr)
                self.vars.append(wi)
                # [ch_out]
                #self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                #for i, p in enumerate(self.vars):
                #    print(i)

            elif name == 'comp_bn':
                # [ch_out]
                #w = nn.Parameter(torch.ones(param[0]))
                #print("!!!---3comp_bn---!!!")
                wr = nn.Parameter(torch.ones(param[0]))
                wi = nn.Parameter(torch.ones(param[0]))
                self.vars.append(wr)#也不归一化
                self.vars.append(wi)
                # [ch_out]
                #self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                #for i, p in enumerate(self.vars):
                #    print(i)

                # must set requires_grad=False
                #running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                #running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                runningr_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                runningr_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([runningr_mean, runningr_var])
                runningi_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                runningi_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([runningi_mean, runningi_var])
                
            elif name == 'multihattention':
                multihead_attn=nn.MultiheadAttention(param[0],param[1])
                self.vars_attn.extend(multihead_attn)
                self.layer_norm1 = nn.LayerNorm(param[0])
                self.dropout1 = nn.Dropout(p=param[2])
            elif name == 'multih_8_attention':
                multihead_attn = nn.MultiheadAttention(param[0], param[1])
                self.vars_attn = multihead_attn
            elif name == 'comp_multihattention':
                self.comp_attn=CollaborativeAttention(param[0], param[0], param[0], param[0], param[1], False, param[2], False, True )

            elif name in ['tanh', 'relu', 'comp_relu', 'comp_softmax','upsample', 'avg_pool2d', 'max_pool2d', 'comp_avg_pool2d', 'comp_max_pool2d',
                          'flatten', 'abs', 'reshape', 'leakyrelu', 'sigmoid','dropout','comp_dropout','softmax','padding','embedding']:
                #这些层不加额外初始化
                continue
            else:
                raise NotImplementedError






    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'comp_conv2d':
                tmp = 'comp_conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name == 'comp_linear':
                tmp = 'comp_linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'comp_avg_pool2d':
                tmp = 'comp_avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'comp_max_pool2d':
                tmp = 'comp_max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'abs', 'tanh', 'relu', 'comp_relu','comp_softmax', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn', 'comp_bn', 'dropout','comp_dropout','softmax','padding','embedding','multihattention', 'comp_multihattention','multih_8_attention']:
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
            vars = self.vars.to(torch.device('cuda'))

        idx = 0
        bn_idx = 0
        #attn_idx=0
        #print("!!!---4all---!!!")
        #for i, p in enumerate(self.vars):
        #    print(i, ": ", self.vars[i])
        #print("!!!---5all_vars---!!!")
        #for i, p in enumerate(vars):
        #    print(i,": ", vars[i])
        #print("-----!!!!!-----")
        #print(self.config)
        for name, param in self.config:
            #print("查看idx")
            #print(name)
            #print(idx)
            name=str(name)
            if name == 'comp_conv2d':
                #wr, br = nn.Parameter(torch.ones(*param[:4], device='cuda')), nn.Parameter(torch.zeros(param[0], device='cuda'))#id=0 w=vars[0] b=vars[1]
                #wi, bi = nn.Parameter(torch.ones(*param[:4], device='cuda')), nn.Parameter(torch.zeros(param[0], device='cuda'))
                #torch.nn.init.kaiming_normal_(wr)
                #torch.nn.init.kaiming_normal_(wi)
                wr, br=vars[idx], vars[idx+2]
                wi, bi=vars[idx+1], vars[idx+3]
                #print("Atention!!")
                #print(len(w[0]))
                #print(idx)
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                #print('in:', x.shape)
                #计算效率改进！
                convr=F.conv2d(x.real, wr, br, stride=param[4], padding=param[5])
                convi=F.conv2d(x.imag, wi, bi, stride=param[4], padding=param[5])
                xr = convr-convi
                xi = convr+convi
                x=xr.type(torch.complex64)+1j*xi.type(torch.complex64)
                idx += 4
                #print('w的size:',w.shape,'\tout:',b.shape)
                #print(name, param, '\tout:', x.shape)
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                #print(name, param, '\tout:', x.shape)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                #print("查看idx")
                #print(name)
                #print(idx)
                x = F.linear(x, w, b)
                idx += 2
                #print('forward:', idx, x.norm().item())
                #print(name, param, '\tout:', x.shape)

            elif name == 'comp_linear':
                #print("!!!!!!!!!!!!!!!!!", x.size())
                #print('---------：', x.shape)
                #wr, br = nn.Parameter(torch.ones(*param, device='cuda')), nn.Parameter(torch.zeros(param[0], device='cuda'))
                #wi, bi = nn.Parameter(torch.ones(*param, device='cuda')), nn.Parameter(torch.zeros(param[0], device='cuda'))
                wr, br=vars[idx], vars[idx+2]
                wi, bi=vars[idx+1], vars[idx+3]
                #torch.nn.init.kaiming_normal_(wr)
                #torch.nn.init.kaiming_normal_(wi)
                #print("查看idx")
                #print(name)
                #print(idx)
                linear_r=F.linear(x.real, wr, br)
                linear_i=F.linear(x.imag, wi, bi)
                xr = linear_r-linear_i
                xi = linear_r+linear_i
                x=xr.type(torch.complex64)+1j*xi.type(torch.complex64)
                idx += 4

            elif name == 'comp_bn':
                #wr, br = nn.Parameter(torch.ones(param[0], device='cuda')), nn.Parameter(torch.zeros(param[0], device='cuda'))
                #runningr_mean, runningr_var = nn.Parameter(torch.zeros(param[0], device='cuda'), requires_grad=False), nn.Parameter(torch.ones(param[0], device='cuda'), requires_grad=False)
                wr, br =vars[idx], vars[idx+2]
                runningr_mean, runningr_var =self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                xr = F.batch_norm(x.real, runningr_mean, runningr_var, weight=wr, bias=br, training=bn_training)
                #wi, bi = nn.Parameter(torch.ones(param[0], device='cuda')), nn.Parameter(torch.zeros(param[0], device='cuda'))
                #runningi_mean, runningi_var = nn.Parameter(torch.zeros(param[0], device='cuda'), requires_grad=False), nn.Parameter(torch.ones(param[0], device='cuda'), requires_grad=False)
                wi, bi = vars[idx+1], vars[idx + 3]
                runningi_mean, runningi_var = self.vars_bn[bn_idx+2], self.vars_bn[bn_idx + 3]
                xi = F.batch_norm(x.imag, runningi_mean, runningi_var, weight=wi, bias=bi, training=bn_training)
                x=xr.type(torch.complex64)+1j*xi.type(torch.complex64)
                idx += 4
                bn_idx += 4
                #print(name, param, '\tout:', x.shape)
            elif name == 'embedding':
                embeds=nn.Embedding(param[0],param[1])
                x=embeds(torch.LongTensor(x))
                #idx += 2
            elif name == 'multihattention':

                xs2 = x.shape[2]
                xs3 = x.shape[3]
                x = x.view(-1, 64, xs2 * xs3)
                x=x.permute(2, 0, 1)#L(S),N,E:target(source) sequence length256, batch size5or1, embedding dimension64
                #print(x.shape)
                #x=x.type(torch.FloatTensor)
                multihead_attn=self.vars_attn
                x,weights= multihead_attn(x, x, x)
                layer_norm1 = self.layer_norm1
                dropout1 = self.dropout1
                x=layer_norm1(x)
                x=dropout1(x)
                #print(name, param, '\tout:', x.shape)
                x=x.permute(1, 2, 0)
                x = x.view(-1, 64, xs2, xs3)
                #print('变形后:', param, '\tout:', x.shape)
                #attn_idx += 1

            elif name == 'multih_8_attention':
                xs2=x.shape[2]
                xs3=x.shape[3]
                x = x.view(-1, 32, xs2*xs3)
                x = x.permute(2, 0, 1)  # L(S),N,E:target(source) sequence length256, batch size5or1, embedding dimension64
                # print(x.shape)
                # x=x.type(torch.FloatTensor)
                multihead_attn = self.vars_attn
                A=x.real
                B=x.imag
                #x=(multihead_attn(A,A,A)[0]-multihead_attn(A,B,B)[0]-multihead_attn(B,A,B)[0]-multihead_attn(B,B,A)[0]).type(torch.complex64)
                #+1j*(multihead_attn(A,A,B)[0]+multihead_attn(A,B,A)[0]+multihead_attn(B,A,A)[0]-multihead_attn(B,B,B)[0]).type(torch.complex64)
                x1,w1=multihead_attn(A,A,A)
                x2,w2=multihead_attn(A,B,B)
                x3,w3=multihead_attn(B,A,B)
                x4,w4=multihead_attn(B,B,A)
                x5,w5=multihead_attn(A,A,B)
                x6,w6=multihead_attn(A,B,A)
                x7,w7=multihead_attn(B,A,A)
                x8,w8=multihead_attn(B,B,B)
                x=(x1-x2-x3-x4).type(torch.complex64)+1j*(x5+x6+x7-x8).type(torch.complex64)
                x = x.permute(1, 2, 0)
                x = x.view(-1, 32, xs2, xs3)

            elif name == 'comp_multihattention':
                xs2 = x.shape[2]
                xs3 = x.shape[3]
                x = x.view(-1, 64, xs2 * xs3)
                x = x.permute(0, 2, 1)
                x=self.comp_attn(x)
                x=x[0]
                #print(x.size())
                x = x.permute(0, 2, 1)
                x = x.view(-1, 64, xs2, xs3)

            elif name == 'flatten':
                #print('flatten前：',x.shape)#无参 全展平了
                x = x.contiguous().view(x.size(0), -1)
                #print("!!!!!!!!!!!!!!!!!",x.size())
                #print('flatten后：',x.shape)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
                #print(name,  '\tout:', x.shape)
            elif name == 'dropout':
                x=F.dropout(x,p=param[0])
                #print(name,'\tout:', x.shape)
            elif name == 'comp_dropout':
                x=complex_dropout(x,param[0])
            elif name == 'softmax':
                x=F.softmax(x,dim=0)
                #print(name, '\tout:', x.shape)
            elif name == 'comp_softmax':
                x=F.softmax(x.real,dim=0).type(torch.complex64)+1j*F.softmax(x.imag,dim=0).type(torch.complex64)
                #print(name, '\tout:', x.shape)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
                #print(name, '\tout:', x.shape)
            elif name == 'comp_relu':
                x=F.relu(x.real).type(torch.complex64)+1j*F.relu(x.imag).type(torch.complex64)
                #改变输入？
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])# kernel_size, stride=None, padding=0
                #print(name, param, '\tout:', x.shape)
            elif name =='comp_max_pool2d':
                x_real = F.max_pool2d(x.real, param[0], param[1], param[2])
                x_img = F.max_pool2d(x.imag, param[0], param[1], param[2])
                x = x_real.type(torch.complex64) + 1j * x_img.type(torch.complex64)
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
                #print(name, param, '\tout:', x.shape)
            elif name =='comp_avg_pool2d':
                x_real = F.avg_pool2d(x.real, param[0], param[1], param[2])
                x_img = F.avg_pool2d(x.imag, param[0], param[1], param[2])
                x = x_real.type(torch.complex64) + 1j * x_img.type(torch.complex64)
            elif name =='padding':
                pad=nn.ZeroPad2d(padding=(param[0],param[0],param[1],param[1]))
                x=pad(x)
                #print(name, param, '\tout:', x.shape)
            elif name == 'abs':
                x=x.abs()
            else:
                print(name)
                raise NotImplementedError

        # make sure variable is used properly
        #assert idx == len(vars)
        #assert bn_idx == len(self.vars_bn)


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
        return self.vars