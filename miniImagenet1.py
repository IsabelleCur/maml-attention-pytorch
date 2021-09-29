import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import pickle

import pandas as pd
from numpy import transpose
#实验一

class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :,
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """
    
    

    def __init__(self, mode, batchsz, n_way, k_shot, k_query, startidx=0):#构造函数
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        #batchsz在两个值之间切换
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.mode=mode
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query' % (
        mode, batchsz, n_way, k_shot, k_query))
        
        Xd = pickle.load(open('2016.04C.multisnr.pkl','rb'), encoding='iso-8859-1')
        snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
        #print(Xd.keys())
        #print(len(Xd.keys()))

        #在j=0，1的情况分别计算，并形成List
        #print(mods)
        #print(len(mods))
        #print(snrs)
        X = [] #输入的sample 2*128的list
        lbl = [] #需要分类的label(mod,snr)
        #for mod in mods:
        #    for j in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:

        #        X.append(Xd[(mod, snrs[j])])
        #        for i in range(Xd[(mod, snrs[j])].shape[0]):  lbl.append((mod, snrs[j]))

        #X = np.vstack(X)
        #X=X.tolist()
        #print("-----!!!!!-----")
        for mod in mods:

            X.append(Xd[(mod, 0)])
            for i in range(Xd[(mod, 0)].shape[0]):
                lbl.append((mod, 0))
                #print(len(Xd[(mod, 0)][i]))#2
                #print(len(Xd[(mod, 0)][i][0]))#128
        X = np.vstack(X)

        #print(X[0])
        #print(X[0][0])
        #print(X.shape) #sample_num*2*128
        #print('label长度')
        #print(len(lbl))

        np.random.seed(2016)
        n_examples = np.array(X).shape[0]
        #print(n_examples)
        n_test = n_examples//4
        #X2=[]
        #lbl2=[]
        #for mod in mods[5:]:
            #for j in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:

                #X2.append(Xd[(mod, snrs[j])])
                #for i in range(Xd[(mod, snrs[j])].shape[0]):  lbl2.append((mod, snrs[j]))

        #X2 = np.vstack(X2)
        #X2=X2.tolist()
        #X=X+X2
        #lbl=lbl+lbl2
        #n_total = np.array(X).shape[0]
        #print(n_total)
        test_idx = np.random.choice(range(0,n_examples), size=n_test, replace=False)
        train_idx = list(set(range(0,n_examples))-set(test_idx))

        #print(len(train_idx))
        #print(len(test_idx))


        X_train = X[train_idx]#还是输入样本的形式
        X_test =  X[test_idx]
        Y_train=np.array(lbl)[train_idx]
        #print(len(lbl))
        Y_test=np.array(lbl)[test_idx]
        #lbl=lbl.toList();

        #X2 = []  # 输入的sample 2*128的list
        #lbl2 = []  # 需要分类的label(mod,snr)

        #X2 = np.vstack(X2)

        #X_train=X_train+X2
        #Y_train=Y_train+lbl2
        
        if(mode=='train'):
            X=X_train;
            lbl=Y_train
        elif(mode=='test'):
            X=X_test
            lbl=Y_test

        #def to_onehot(yy):
        #    yy1 = np.zeros([len(yy), max(yy)+1])
        #    yy1[np.arange(len(yy)),yy] = 1
        #    return yy1
        #Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
        #Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))#转换成独热向量编码

        #print(len(X_train))
        #print(len(Y_train))
        #print(len(X_test))
        #print(len(Y_test))
        #print(len(X_test[0]))
        #print(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
        #print(len(Y_test[0]))

        in_shp = list(X_train.shape[1:])
        #print(X_train.shape)
        #print(in_shp)
        #print([1]+in_shp)
        classes = mods
    
        #if mode == 'train':
        #    self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
        #                                         transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
        #                                         transforms.ToTensor(),
        #                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #                                         ])
        #else:
        #    self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
        #                                         transforms.Resize((self.resize, self.resize)),
        #                                         transforms.ToTensor(),
        #                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #                                         ])

        #self.path = os.path.join(root, 'images')  # image path
        csvdata = self.loadCSV(lbl,X)  # 形成csv path，调用loadCSV，返回map
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]] 每类有一个List data的格式是必要的吗？预处理radio
            self.img2label[k] = i + self.startidx  # {(Str)mod:label}
        self.cls_num = len(mods)#类别的数量
        #print("类别的数量")
        #print(len(mods))

        self.create_batch(self.batchsz)

    def loadCSV(self,a_lbl,a_X ):#map:class->属于改class的所有2*128组成的列表
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        for i in range(len(a_lbl)):
            samples = a_X[i]
            
            label0 = a_lbl[i][0]#应该改对了
            # append filename to current label
            if label0 in dictLabels.keys():
                dictLabels[label0].append(samples)
            else:
                dictLabels[label0] = [samples]#samples:[[128],[128]]
            #print('测试')    
            #print(len(dictLabels))
        return dictLabels

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        self.support_y_batch=[];
        self.query_y_batch=[];
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []#函数内的局部变量 用于计算support_x_batch
            support_y=[]
            query_y=[]
            support_xy=[]
            query_xy=[]
            #if self.mode=='test':
                #for i in range(5):
                    #selected_cls[i]=i
            #print("测试")
            #print(len(self.data))
            #print(len(self.data[0]))
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest]

                support_y_temp=np.ones(self.k_shot)*cls;
                query_y_temp=np.ones(self.k_query)*cls;
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain 每类有一个list是被选中的该类的K_shot个图片样本，共有n_way项 5*1(15)*2*128
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())
                support_y.append(support_y_temp);
                query_y.append(query_y_temp);
            #random.shuffle(selected_cls)

            for i in range(len(support_x)):
                temp_xy=[support_x[i],support_y[i]]
                support_xy.append(temp_xy)

            for j in range(len(query_x)):
                t_xy=[query_x[j],query_y[j]]
                query_xy.append(t_xy)

            #print(np.array(query_xy[0][0]).shape)
            # shuffle the correponding relation between support set and query set
            random.shuffle(support_xy)
            random.shuffle(query_xy)
            support_xy=np.array(support_xy).transpose(1,0,2).tolist()
            query_xy=np.array(query_xy).transpose(1,0,2).tolist()
            #print('打印尺寸')
            #print(np.array(query_xy[0]).shape)
            #print(np.array(query_xy[1]).shape)
            #print(np.array(query_x).shape)
            #print(len(query_y))
            #print(np.array(query_xy).shape)
            #print(np.array(query_xy[0]).shape)
            #print(query_xy[1][0].shape)

            self.support_x_batch.append(support_xy[0])  # append set to current sets有batchsz个这样的list
            self.query_x_batch.append(query_xy[0])  # append sets to current sets
            self.support_y_batch.append(support_xy[1])
            self.query_y_batch.append(query_xy[1])
            #createbatch为self（此类）创建了两个list:batch并赋值gg
        #print('打印尺寸')
        #print(np.array(self.support_x_batch).shape)

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        #print('get_item好像没进来')
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz,1, 2, 128)#函数内的局部变量 但是会返回 比较有用
        # [setsz]
        
        #print(len(support_x))
        #print(len(support_x[0]))
        #print(len(support_x[0][0]))
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz,  1, 2, 128)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)

        support_x = [[item]
                             for sublist in self.support_x_batch[index] for item in sublist]#for support_x中每个类的sublist，for  sublist中的每个元素，即每个完整路径名
        support_y = np.array(
            [item  # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_y_batch[index] for item in sublist]).astype(np.int32)#对应到每一个support_x的样本的y值，即label值

        query_x = [[item]
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([item
                            for sublist in self.query_y_batch[index] for item in sublist]).astype(np.int32)

        #for i, path in enumerate(support_x0):
         #   support_x[i][0]=torch.FloatTensor(path)

        #for i, path in enumerate(query_x0):
        #    query_x[i][0]=torch.FloatTensor(path)

        #print('尺寸')
        #print(np.array(support_x0).shape)
        #print(np.array(support_x).shape)
        #print((np.array(support_x0)==np.array(support_x)).all())
        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted

        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx
        #把label顺序打乱
        #不需要x和y对应
        # print('relative:', support_y_relative, query_y_relative)

        #for i, path in enumerate(flatten_support_x):
        #    support_x[i] = self.transform(path)#3*168*168张量

        #for i, path in enumerate(flatten_query_x):
        #    query_x[i] = self.transform(path)
        # print(support_set_y)
        # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)
        
        #print(len(support_x))
        #print(len(support_x[0]))
        #print(len(support_x[0][0]))
        #print(len(support_x[0][0][0]))
        #print(np.array(query_x).shape)#    5*1*2*128  ->  75*1*2*128
        #print(np.array(torch.LongTensor(query_y_relative)).shape)
        return np.array(support_x), np.array(torch.LongTensor(support_y_relative)), np.array(query_x), np.array(torch.LongTensor(query_y_relative))

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion()

    tb = SummaryWriter('runs', 'mini-imagenet')
    

    tb.close()

    
