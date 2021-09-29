import torch, os
import numpy as np
from miniImagenet0 import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys
import argparse
import pickle
# import torchsnooper
from complexmeta import Meta

# gpu_id="1,2,3" ; #指定gpu id
# 配置环境  也可以在运行时临时指定 CUDA_VISIBLE_DEVICES='2,7' Python train.py
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 这里的赋值必须是字符串，list会报错

# device_ids=range(torch.cuda.device_count())  #torch.cuda.device_count()=2


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


# @torchsnooper.snoop()
def main():
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        # ('reshape',[1,2,128]),
        ('comp_conv2d', [32, 1, 1, 1, 1, 0]),
        # ('relu', [True]),
        # ('bn', [64]),
        # ('conv2d', [512, 1024, 1, 1, 1, 0]),

        ('comp_conv2d', [32, 32, 1, 3, 1, 0]),
        ('comp_relu', [True]),
        ('comp_bn', [32]),
        # ('max_pool2d', [2, 2, 0]),
        ('comp_conv2d', [32, 32, 1, 3, 1, 0]),
        ('comp_relu', [True]),
        ('comp_bn', [32]),
        #('max_pool2d', [2, 2, 2]),

        ('multih_8_attention', [32, 8]),

        ('comp_conv2d', [32, 32, 1, 3, 1, 0]),
        ('comp_relu', [True]),
        ('comp_bn', [32]),#少两个

        ('comp_conv2d', [32, 32, 1, 3, 1, 0]),
        ('comp_relu', [True]),
        ('comp_bn', [32]),#少一个

        ('flatten', []),
        #('flatten', []),
        #('comp_linear', [512, 63488]),  # output input
        ('comp_linear', [args.n_way, 7680]),
        ('abs', [])#,
        #('softmax',[])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    # maml = torch.nn.DataParallel(maml)  # 前提是model已经.cuda() 了
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in maml.parameters())))

    # batchsz here means total episode number
    mini = MiniImagenet(mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=4000)

    mini_test = MiniImagenet(mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100)

    for epoch in range(args.epoch // 4000):
        # fetch meta_batch sz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
        # Dataset,batch_zsize,shuffle,num_workers
        # for each in mini:
        #    print(each)
        # print('打印db')
        # print(db[0])
        # print('测试')
        # print(len(db))
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):  # 从get_item出来
            # print(len(x_spt))
            # print(len(x_spt[0]))
            # print(len(x_spt[0][0]))
            # print(len(x_spt[0][0][0]))
            # print(len(x_spt[0][0][0][0]))
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device).type(torch.complex64), y_spt.to(device), x_qry.to(device).type(torch.complex64), y_qry.to(device)
            # print("x_spt_BEFORE: ", x_spt[0][0][0][0][0])
            accs = maml(x_spt, y_spt, x_qry, y_qry)
            torch.save(maml, 'complex8atten32_5shot_SGNL.pkl')  # 保存整个神经网络的结构和模型参数

            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs)

            if step % 500 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device).type(torch.complex64), y_spt.squeeze(0).to(
                        device), \
                                                 x_qry.squeeze(0).to(device).type(torch.complex64), y_qry.squeeze(0).to(
                        device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=330000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=2)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()
