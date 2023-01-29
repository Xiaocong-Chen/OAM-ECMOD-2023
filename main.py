import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as Data

import numpy as np
from zmq import device

from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler

import random
from ctypes import util
import logging

import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
from itertools import combinations, permutations

import torchvision
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D
from torchvision.datasets import MNIST

import sys
from PIL import Image
torch.manual_seed(1)    # reproducible
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class MyDataset(Dataset):
    def __init__(self, root: str, dataset_name: str, num_neibs: int = 8, fea_start: int = -1, fea_end: int = -1):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name

        od_data = np.loadtxt(root + dataset_name + '.txt', delimiter=',')
        xa = od_data[:,0:-1]
        xs = None
        if fea_start == -1 and fea_end == -1:
            xs = od_data[:, 0:-1]
        else:
            xs = od_data[:, fea_start: fea_end]
        self.X_train = xs.astype(np.float32)
        labels = od_data[:, -1]
        labels = labels.astype(np.float32)

        self.labels = labels
        self.size = labels.shape[0]
        self.data_all = torch.tensor(xa.astype(np.float32), dtype=torch.float32)
        self.data = torch.tensor(self.X_train, dtype=torch.float32)
        # self.data = torch.tensor(self.X_train)
        self.targets = torch.tensor(labels, dtype=torch.float32)
        # self.targets = torch.tensor(labels)

        # a neighbor list for each view
        self.num_views = NUM_VIEWS
        self.num_neibs = num_neibs
        self.neibs = []
        self.neibs_global = []
        self.neibs_local = []
        self.weights_global = []
        
        self.noneibs = []
        self.noneibs_global = []
        self.neibs_local = []
        self.noneibs_local = []
        
        
        self.init_knn()

    def get_name(self):
        return self.dataset_name

    def get_all(self):
        return self.data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        
        Returns:
            tuple: (sample, target, index)
        """
        input_tensor_all = self.data_all[index]
        sample, target = self.data[index], (self.targets[index])
        # input_tensor_each_view = []
        # each_view_len = int(sample.shape[0] / self.num_views) 
        # for i in range(self.num_views):
        #     l, r = int(i * each_view_len),int((i + 1) * each_view_len)
        #     if i != self.num_views - 1:
        #         x = sample[l:r]
        #     else:
        #         x = sample[l:]
        #     input_tensor_each_view.append(x)
        # input_tensor_each_view = sample
        input_tensor_each_view_couple1 = sample
        input_tensor_each_view_couple2 = sample
        
        
        # if index == 0:
        #     tmp = self.neibs[index]
        #     list.sort(tmp)
        #     print(tmp)
        if len(self.neibs_global) > 0:
            tmp = random.randint(0, len(self.neibs_global[index]) - 1)
            id_global_neighbor = self.neibs_global[index][tmp]
            weight_global = self.weights_global[index][tmp]
            id_local_neighbor = random.choice(self.neibs_local[index])
        else:
            id_global_neighbor = index
            id_local_neighbor = index
            weight_global = 0.0
        weight_global = np.float32(weight_global) # 权重矩阵 A
        neighbor_global, neighbor_local = self.data[id_global_neighbor], self.data[id_local_neighbor]
        
        # 
        
        return sample, neighbor_global, neighbor_local, weight_global, target, index, id_global_neighbor, id_local_neighbor, \
               input_tensor_all, input_tensor_each_view_couple1, input_tensor_each_view_couple2
    
    def __len__(self):
        return len(self.data)

    def init_knn(self):
        X = self.X_train
        nbrs = NearestNeighbors(n_neighbors=self.num_neibs+1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        # print(indices)
        for i in range(self.size):
            self.neibs.append((list(indices[i, 1: ])))
        temp_n = set(range(self.size))
        for i in range(self.size):
            self.noneibs.append(list(temp_n - set(self.neibs[i])))
        print(self.neibs[0], self.noneibs[0])
        return indices

    def update_knn(self):
        return

    def get_knn(self):
        return self.neibs

    def set_knn(self, neibs_global, neibs_local, weights_global):
        self.neibs_global = neibs_global
        self.noneibs_global = []
        temp_n = set(range(self.size))
        for i in range(self.size):
            self.noneibs_global.append(list(temp_n - set(self.neibs_global[i])))
        self.weights_global = weights_global
        self.neibs_local = neibs_local
        self.noneibs_local = []
        temp_n = set(range(self.size))
        for i in range(self.size):
            self.noneibs_local.append(list(temp_n - set(self.neibs_local[i])))
        return

    def get_labels(self):
        return self.labels

    def _check_exists(self):
        return os.path.exists(self.data_file)
    
data_root_path = 'data_new/'
METHOD_NAME = None
RECORD_FILE_NAME = None
DATA_DIM = 16
OUTLIER_RATE = 3 / 100
DATAID = -1
NUM_VIEWS = 2
FILE_NAME = 'zoo-101-2-0.05-0.05-0.05'

def get_view_fea():
    avg_num = int(DATA_DIM / NUM_VIEWS)
    left_num = DATA_DIM % NUM_VIEWS
    fea_num = [avg_num for a in range(NUM_VIEWS)]
    for i in range(left_num):
        fea_num[NUM_VIEWS - 1 - i] += 1
    start = 0
    end = 0
    s = [0 for a in range(NUM_VIEWS)]
    e = [0 for a in range(NUM_VIEWS)]
    for v in range(NUM_VIEWS):
        end = start + fea_num[v]
        s[v] = start
        e[v] = end
        start = end
    return s, e

def set_num_views(n):
    global NUM_VIEWS
    NUM_VIEWS = n

def set_record_file(f):
    global RECORD_FILE_NAME
    RECORD_FILE_NAME = f

def set_data_id(i):
    global DATAID
    DATAID = i

def set_file_name(f):
    global FILE_NAME
    FILE_NAME = f

def set_method_name(n):
    global METHOD_NAME
    METHOD_NAME = n

def set_dim(d):
    global DATA_DIM
    DATA_DIM = d

def set_outlier_ratio(p):
    global OUTLIER_RATE
    OUTLIER_RATE = p

def compute_auc(labels, scores):
    return roc_auc_score(labels, scores)

def get_percentile(scores, threshold):
    per = np.percentile(scores, 100 - int(100 * threshold))
    return per

def compute_f1_score(labels, scores):
    per = get_percentile(scores, OUTLIER_RATE)
    y_pred = (scores >= per)
    # print(np.sum(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(labels.astype(int),
                                                               y_pred.astype(int),
                                                               average='binary')
    return precision, recall, f1

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if not torch.cuda.is_available():
    device = 'cpu'
else:
    device = 'cuda'

device = 'cuda'
# AutoEncoder
class Autoencoder(nn.Module):
    def __init__(self, data_dim):
        super(Autoencoder, self).__init__()
        self.rep_dim = 32
        self.data_dim = data_dim

        en_layers_num = None
        if DATA_DIM == 16:
            # en_layers_num[输入层大小、隐藏层大小、输出层大小]
            en_layers_num = [self.data_dim, 128, self.rep_dim] # zoo
        elif DATA_DIM == 13:
            en_layers_num = [self.data_dim, 128, self.rep_dim] # wine
        elif DATA_DIM == 30:
            en_layers_num = [self.data_dim, 128, self.rep_dim] # wdbc
        elif DATA_DIM == 8:
            en_layers_num = [self.data_dim, 128, self.rep_dim] # pima, peast
        elif DATA_DIM == 16:
            en_layers_num = [self.data_dim, 128, self.rep_dim] # letter
        elif DATA_DIM == 400:
            en_layers_num = [self.data_dim, 128, self.rep_dim] # speech
        elif DATA_DIM == 64:
            en_layers_num = [self.data_dim, 128, self.rep_dim] # optitigits
        elif DATA_DIM == 10:
            en_layers_num = [self.data_dim, 128, self.rep_dim] # cover
            
        self.encoder = nn.Sequential(nn.Linear(en_layers_num[0], en_layers_num[1]), 
                                     nn.Tanh(), 
                                     nn.Linear(en_layers_num[len(en_layers_num)-2],en_layers_num[len(en_layers_num)-1]))
        de_layers_num = list(reversed(en_layers_num))
        self.decoder = nn.Sequential(nn.Linear(de_layers_num[0], de_layers_num[1]), 
                                     nn.Tanh(),
                                     nn.Linear(de_layers_num[len(de_layers_num)-2],de_layers_num[len(de_layers_num)-1]), 
                                     nn.Sigmoid())
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
# MLP
class MLP4Views(nn.Module):
    def __init__(self, data_dim, data_all_dim):
        super(MLP4Views, self).__init__()
        self.data_all_dim = data_all_dim
        self.data_dim = data_dim
        # each_view_len = []
        # tmp = int(self.data_dim / NUM_VIEWS)
        # for i in range(NUM_VIEWS):
        #     if(i == NUM_VIEWS-1):
        #         each_view_len.append(tmp + self.data_dim % NUM_VIEWS)
        #     else:
        #         each_view_len.append(tmp)
        self.w1_common = nn.Parameter(torch.randn(self.data_all_dim, 80))
        self.b1_common = nn.Parameter(torch.tensor(0.1)) # 80
        self.w2_common = nn.Parameter(torch.randn(80, 32))
        self.b2_common = nn.Parameter(torch.tensor(0.1)) # 32
        self.w1_each_view = nn.Parameter(torch.randn(self.data_dim, 80))
        self.b1_each_view = nn.Parameter(torch.tensor(0.1)) # each 80
        self.w2_each_view = nn.Parameter(torch.randn(80, 32))
        self.b2_each_view = nn.Parameter(torch.tensor(0.1)) # each 32
    
    def forward(self, input_tensor_all, input_tensor_each_view, input_tensor_each_view_couple1, input_tensor_each_view_couple2):
        
        layer1_common = torch.sigmoid(torch.matmul(input_tensor_all, self.w1_common) + self.b1_common)
        layer2_common = torch.sigmoid(torch.matmul(layer1_common, self.w2_common) + self.b2_common)
        
        layer1_each_view = torch.sigmoid(torch.matmul(input_tensor_each_view, self.w1_each_view) + self.b1_each_view)
        layer1_each_view_couple1 = torch.sigmoid(torch.matmul(input_tensor_each_view_couple1, self.w1_each_view) + self.b1_each_view)
        layer1_each_view_couple2 = torch.sigmoid(torch.matmul(input_tensor_each_view_couple2, self.w1_each_view) + self.b1_each_view)
        
        layer2_each_view = torch.sigmoid(torch.matmul(layer1_each_view, self.w2_each_view) + self.b2_each_view)
        layer2_each_view_couple1 = torch.sigmoid(torch.matmul(layer1_each_view_couple1, self.w2_each_view) + self.b2_each_view)
        layer2_each_view_couple2 = torch.sigmoid(torch.matmul(layer1_each_view_couple2, self.w2_each_view) + self.b2_each_view)
        
        # 特定视图的表示,The representations of samples in this latent intact space are learned jointly by a common neural network and V view-specific neural networks
        offset_each_view = layer2_each_view # 每个视图的偏置
        middle_layer_each_view = layer2_common + layer2_each_view # 共享视图+每个视图的偏置
        middle_layer_each_view_couple1 = layer2_common + layer2_each_view_couple1 # 共享视图+每个视图的偏置，当作某个视图中样本的映射
        middle_layer_each_view_couple2 = layer2_common + layer2_each_view_couple2 # 共享视图+每个视图的偏置，当作某个视图中样本的映射
        
        return layer2_common, offset_each_view, middle_layer_each_view, middle_layer_each_view_couple1, middle_layer_each_view_couple2

# modle
class My_model(nn.Module):
    def __init__(self,data_dim,data_all_dim):
        super(My_model, self).__init__()
        self.data_dim = data_dim
        self.ae = Autoencoder(data_dim)
        self.mlp4views = MLP4Views(data_dim,data_all_dim)
        
    # x1:input_tensor_all, x2:input_tensor_each_view, x3:input_tensor_each_view_couple1, x4:input_tensor_each_view_couple2, x5:w1_common, x6:b1_common, x7:w2_common, x8:b2_common, x9:w1_each_view, x10:b1_each_view, x11:w2_each_view, x12:b2_each_view
    def forward(self,x,x1,x2,x3,x4): 
        # x = [16, 128, 32]  >> encode and decode
        encoded, decoded = self.ae(x) 
        # y1:layer2_common, y2:offset_each_view, y3:middle_layer_each_view, y4:middle_layer_each_view_couple1, y5:middle_layer_each_view_couple2
        y1, y2, y3, y4, y5 = self.mlp4views(x1,x2,x3,x4)
        return encoded, decoded, y1, y2, y3, y4, y5
    
class ATrainer:
    def __init__(self, optimizer_name: str = 'adam', 
                 lr: float = 0.001, 
                 n_epochs: int = 150,
                 lr_milestones: tuple = (), 
                 batch_size: int = 128, 
                 weight_decay: float = 1e-6, 
                 device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader


    def train(self, id_round, id_view, dataset, view_net, view_net1, pre_train, pre_epochs, module_weight, albation_set):
        logger = logging.getLogger()
        
        # Set device for network
        view_net = view_net.to(self.device)
        view_net1 = view_net1.to(self.device)
        
        # Get train data loader
        train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.n_jobs_dataloader, drop_last=True)

        # Set optimizer (Adam optimizer for now)
        all_parameters1 = view_net.parameters()
        all_parameters2 = view_net1.parameters()
        
        optimizer1 = optim.Adam(all_parameters1, lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')
        optimizer2 = optim.Adam(all_parameters2, lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=self.lr_milestones, gamma=0.1)
        scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=self.lr_milestones, gamma=0.1)

        # Training
        start_time = time.time()
        view_net.train()
        view_net1.train()

        num_epochs = self.n_epochs
        if pre_train and id_round == 0:
            num_epochs = pre_epochs

        for epoch in range(num_epochs):
            # time.sleep(1)
            # if epoch in self.lr_milestones:
            #     logger.info('  LR scheduler: new learning rate is %g' % float(scheduler1.get_lr()[0]))
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, neighbor_global, neighbor_local, weights_global, labels, index, index_g, index_l, \
                    input_tensor_all, input_tensor_each_view_couple1, input_tensor_each_view_couple2 = data
                
                inputs = inputs.to(self.device)
                neighbor_global = neighbor_global.to(self.device)
                neighbor_local = neighbor_local.to(self.device)
                weights_global = weights_global.to(self.device)
                labels = labels.to(self.device)
                
                input_tensor_all = input_tensor_all.to(self.device)
                input_tensor_each_view_couple1 = input_tensor_each_view_couple1.to(self.device)
                input_tensor_each_view_couple2 = input_tensor_each_view_couple2.to(self.device)
                
                # Zero the network parameter gradients
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                objectsize = inputs.shape[0]
                encoded, outputs = view_net(inputs)
                encoded_global_neighbor, _ = view_net(neighbor_global)
                # encoded_local_neighbor, _ = view_net(neighbor_local)
                # 带视图间关系的特定视图的表示
                middle_y_common, middle_y_each_view, offset_each_view, middle_y_each_view_couple1, middle_y_each_view_couple2 = \
                    view_net1(input_tensor_all, inputs, input_tensor_each_view_couple1, input_tensor_each_view_couple2)
                
                # contrast_loss
                con_loss = self.SSL(encoded, middle_y_each_view)
                
                # intra loss，保持原始视图空间内样本间的结构，样本间的关系如下：
                intra_loss = torch.abs(torch.mean(torch.sum((middle_y_each_view_couple2 - middle_y_each_view_couple1) ** 2, dim=tuple(range(1, encoded.dim())))) - \
                    torch.mean(torch.sum((input_tensor_each_view_couple2 - input_tensor_each_view_couple1) ** 2, dim=tuple(range(1, inputs.dim())))))
                
                # reconstruction loss
                recon_error = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                # the neighbors' distances
                dis_global = torch.sum((encoded - encoded_global_neighbor) ** 2, dim=tuple(range(1, outputs.dim())))
                # dis_local = torch.sum((encoded - encoded_local_neighbor) ** 2, dim=tuple(range(1, outputs.dim())))
                
                # loss_total = recon_error + 0.05*con_loss
                # loss_total = recon_error + intra_loss
                loss_total = recon_error + 0.05*con_loss + intra_loss
                # 最开始没有G，预先还没训练，
                if pre_train and id_round == 0:
                    scores = loss_total
                else:
                    # module_weight：模型权重,  weights_global： Gauss kernel based affinity matrix带权重的共识相邻矩阵来最小化邻居距离
                    scores = albation_set[0] * loss_total + albation_set[1] * module_weight * weights_global * dis_global

                loss = torch.mean(scores)
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                scheduler1.step()
                scheduler2.step()
                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  View {} Round {} Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f} '
                        .format(id_view + 1, id_round, epoch + 1, num_epochs, epoch_train_time, loss_epoch / n_batches))
            # if epoch in np.array([10 - 1, 50 - 1, 100 - 1, 150 - 1, 200 - 1, 250 - 1, 300 - 1,
            #                       350 - 1, 400 - 1, 450 - 1, 500 - 1, 550 - 1, 600 - 1]):

            # logger.info('  Intermediate Test at Epoch {}'.format(epoch + 1))

        view_encoded, recon_error = self.test(id_view, dataset, view_net, view_net1)

        train_time = time.time() - start_time
        logger.info('View {} Training time: {:.3f}'.format(id_view + 1, train_time))
        # logger.info('Finished training.')

        return view_net, view_net1, view_encoded, recon_error

    def test(self, id_view, dataset, ae_net, mlp_net):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)
        mlp_net = mlp_net.to(self.device)

        # Get test data loader
        test_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.n_jobs_dataloader, drop_last=False)

        # Testing
        logger.info('View {} Testing lg_ae...'.format(id_view + 1))
        loss_epoch = 0.0

        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        mlp_net.eval()
        encoded_data = None

        loss_recon_epoch = 0.0

        with torch.no_grad():
            for data in test_loader:
                inputs, _, _, _, labels, idx, _, _, input_tensor_all, input_tensor_each_view_couple1, input_tensor_each_view_couple2 = data
                inputs = inputs.to(self.device)
                input_tensor_all = input_tensor_all.to(self.device)
                input_tensor_each_view_couple1 = input_tensor_each_view_couple1.to(self.device)
                input_tensor_each_view_couple2 = input_tensor_each_view_couple2.to(self.device)

                encoded, outputs = ae_net(inputs)
                middle_y_common, middle_y_each_view, offset_each_view, middle_y_each_view_couple1, middle_y_each_view_couple2 = \
                    mlp_net(input_tensor_all, inputs, input_tensor_each_view_couple1, input_tensor_each_view_couple2)
                
                recon_error = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                con_loss = self.SSL(encoded, middle_y_each_view)
                intra_loss = torch.abs(torch.mean(torch.sum((middle_y_each_view_couple2 - middle_y_each_view_couple1) ** 2, dim=tuple(range(1, encoded.dim())))) - \
                    torch.mean(torch.sum((input_tensor_each_view_couple2 - input_tensor_each_view_couple1) ** 2, dim=tuple(range(1, inputs.dim())))))
                # print(recon_error.size())
                # scores = recon_error + 0.05*con_loss
                # scores = recon_error + intra_loss
                scores = recon_error + intra_loss + 0.05*con_loss
                
                loss = torch.mean(scores)
                loss_recon = torch.mean(recon_error)

                if n_batches == 0:
                    encoded_data = encoded.cpu().data.numpy()
                else:
                    encoded_data = np.concatenate((encoded_data, encoded.cpu().numpy()), axis=0)
                # print(encoded_data.shape)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                loss_recon_epoch += loss_recon.item()
                n_batches += 1

        logger.info('View {}: Test set score: {:.8f}, Recon_score: {:.8f}'
                    .format(id_view, loss_epoch / n_batches, loss_recon_epoch / n_batches))

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)


        # if ae_net.rep_dim == 2:
        #     self.circle_plot(labels, encoded_data, epoch, dataset.get_name())

        # self.dist_plot(labels, scores, epoch, dataset.get_name())

        return encoded_data, scores

    def circle_plot(self, labels, encoded_data, epoch, dataset_name):
        # embedd_dim = 2
        plot_x = encoded_data
        plot_center = np.mean(encoded_data, 0)
        outlier_num = np.sum(labels)
        inlier_num = encoded_data.shape[0] - outlier_num
        print(encoded_data.shape, inlier_num, outlier_num)
        print(plot_center)

        plt.rcParams['figure.figsize'] = (8.0, 8.0)
        plt.scatter(encoded_data[0: inlier_num, 0], encoded_data[0: inlier_num, 1], c='g', marker='*')
        plt.scatter(encoded_data[inlier_num: encoded_data.shape[0], 0],
                    encoded_data[inlier_num: encoded_data.shape[0], 1], c='r', marker='x')
        plt.scatter(plot_center[0], plot_center[1], c='b')
        plt.savefig('./circles_changes/' + dataset_name + '-' + 'lgae' + str(epoch) + '.jpg')
        # plt.show()
        plt.close()


    def dist_plot(self, lables, scores, epoch, dataset_name):
        out_scores = []
        in_scores = []
        for i in range(len(lables)):
            if lables[i] == 1:
                out_scores.append(scores[i])
            else:
                in_scores.append(scores[i])

        plt.figure(figsize=(7, 6))

        font1 = {'family': 'Times New Roman',
                 'style': 'normal',
                 'weight': 'normal',
                 'size': 35,
                 }
        plt.grid(axis='both', linestyle='--')
        plt.hist(in_scores, bins=30, normed=True, facecolor='green', alpha=0.75, label='Inlier')
        plt.hist(out_scores, bins=30, normed=True, facecolor='red', alpha=0.75, label='Outlier')
        # plt.xlabel('Smarts', font1)
        # plt.ylabel('Probability', font1)
        # plt.xticks([])
        # plt.yticks([])
        plt.legend(prop=font1)
        plt.savefig('./loss_changes/' + dataset_name + '-' + 'good' + str(epoch) + '.jpg')
        # plt.close()
        # plt.show()
        return
    
    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        # 改变行和列构造负例  单个负例
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = score(sess_emb_hgnn, sess_emb_lgcn)
        neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
        one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1) # 填充1
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss

def run(data_path, dataset_name, batch_size, learning_rate, k_neibs, module_weight, albation_set):
    logger = logging.getLogger()

    print('#########  LocalAE conducted on ' + dataset_name)
    # print(torch.cuda.get_device_name(0))
    num_rounds = 10
    num_epochs = 16
    
    # 主干网络为 AE + KNN
    net_name = 'AE_KNN'
    
    # 打印日志文件
    logging.basicConfig(level=logging.INFO, filename='running_log')
    
    # 输出格式，
    # 字符串形式的当前时间。默认格式是 “2003-07-08 16:49:45,896”
    # Logger的名字
    # 文本形式的日志级别
    # 用户输出的消息
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    device = 'cuda'
    # 全局的视图数量
    num_views = NUM_VIEWS
    # 邻居节点的个数
    num_neibs = k_neibs
    
    pre_train = False
    pre_epochs = 100

    s, e = get_view_fea()
    
    data_dim = e[-1] - s[0]
    
    all_nets = [None for a in range(num_views)]
    all_nets1 = [None for a in range(num_views)]
    # 数据集
    datasets = [None for a in range(num_views)]
    # 编码后的视图
    view_encoded = [None for a in range(num_views)]
    # 视图重构误差
    recon_error = [None for a in range(num_views)]
    
    labels = None

    for i in range(num_views):
        datasets[i] = MyDataset(root=data_path, dataset_name=dataset_name, num_neibs=num_neibs, fea_start=s[i], fea_end=e[i])
    if labels is None:
        labels = datasets[0].get_labels()

    for v in range(num_views):
        # all_nets[v].set_network(net_name, e[v] - s[v])
        # total_model = My_model(e[v] - s[v])
        # encoded, decoded, y1, y2, y3, y4, y5 = total_model()
        all_nets[v] = Autoencoder(e[v] - s[v])
        all_nets1[v] = MLP4Views(e[v] - s[v], data_dim)
    # print(all_nets1[v].parameters)
    

    Trainer = ATrainer(optimizer_name='adam',
                        lr=learning_rate,
                        n_epochs=num_epochs,
                        lr_milestones=tuple(()),
                        batch_size=batch_size,
                        weight_decay=1e-6,
                        device=device,
                        n_jobs_dataloader=0)

    for eround in range(num_rounds):
        for id_view in range(num_views):
            all_nets[id_view], all_nets1[id_view], view_encoded[id_view], recon_error[id_view] =\
                Trainer.train(eround, id_view, datasets[id_view], all_nets[id_view], all_nets1[id_view], pre_train, pre_epochs,
                              module_weight, albation_set)
        # 利用前面网络学到的参数初始化共识邻域矩阵
        neibs_global, neibs_local, weights_global = get_new_knn2(view_encoded, num_neibs)

        print('round %d: ' % eround)
        for id_view in range(num_views):
            tmp1 = list(neibs_global[id_view][0])
            tmp2 = list(neibs_local[id_view][0])
            list.sort(tmp1)
            list.sort(tmp2)
            print('View %d: global neighbors is %s, local neighbors is %s.' % (id_view + 1, tmp1, tmp2))
            datasets[id_view].set_knn(neibs_global[id_view], neibs_local[id_view], weights_global[id_view])

    recon_scores, knn_scores = cal_scores(view_encoded, num_neibs, recon_error)

    recon_auc = compute_auc(labels, recon_scores)
    # recon_fs = compute_f1_score(labels, recon_scores)
    # auc, precision, recall, f1-score
    print('recon performance: %.4f  ' % (recon_auc))
    # print('recon performance: %.4f  %.4f  %.4f  %.4f' % (recon_auc, recon_fs[0], recon_fs[1], recon_fs[2]))

    knn_auc = compute_auc(labels, knn_scores)
    # knn_fs = compute_f1_score(labels, knn_scores)
    # auc, precision, recall, f1-score
    print('knn performance: %.4f  ' % (knn_auc))
    # print('knn performance: %.4f  %.4f  %.4f  %.4f' % (knn_auc, knn_fs[0], knn_fs[1], knn_fs[2]))

    # Scale to range [0,1]
    total_scores = cal_final_scores(recon_scores, knn_scores)
    total_auc = compute_auc(labels, total_scores)
    # total_fs = compute_f1_score(labels, total_scores)
    # auc, precision, recall, f1-score
    print('total performance: %.4f ' % (total_auc))
    # print('total performance: %.4f  %.4f  %.4f  %.4f' % (total_auc, total_fs[0], total_fs[1], total_fs[2]))

    test_auc = albation_set[0] * recon_auc + albation_set[1] * knn_auc
    # test_f1 = albation_set[0] * recon_fs[0] + albation_set[1] * knn_fs[0]

    if albation_set == (1, 1):
        test_auc =  knn_auc
        # test_auc = total_auc
        # test_f1 = knn_fs[0]
    return test_auc
    # return test_auc, test_f1
    # return knn_auc, knn_fs[0]


def cal_final_scores(recon_scores, knn_scores):
    # total_scores = recon_scores + knn_scores
    s1 = (recon_scores - np.min(recon_scores)) / (np.max(recon_scores) - np.min(recon_scores))
    s2 = (knn_scores - np.min(knn_scores)) / (np.max(knn_scores) - np.min(knn_scores))
    total_scores = 0.5*s1 + s2
    # total_scores = np.maximum(s1, s2)
    # rank-based
    return total_scores


def get_new_knn2(view_encoded, num_neibs):
    num_views = len(view_encoded)
    num_obj = view_encoded[0].shape[0]

    neibs_local = [[] for a in range(num_views)]
    kth_neib_local = [[] for a in range(num_views)]
    for i in range(num_views):
        # 想为数据集Y中的每一个点，在数据集X中找到距其（y）最近的k个点.
        nbrs = NearestNeighbors(n_neighbors=num_neibs + 1, algorithm='ball_tree').fit(view_encoded[i]) # X
        distances, indices = nbrs.kneighbors(view_encoded[i]) # Y
        for j in range(num_obj):
            indice_j = list(indices[j])
            if j in indice_j:
                indice_j.remove(j)
            else:
                indice_j = indice_j[1: ]
            neibs_local[i].append(indice_j)
            kth_neib_local[i].append(indices[j, num_neibs])

    weights_global = [[] for a in range(num_views)]
    neibs_global = [[] for a in range(num_views)]
    for i in range(num_views):
        for j in range(num_obj):
            tmp = []
            tmp_weights = []
            for k in range(num_views):
                if k != i:
                    tmp = tmp + neibs_local[k][j]
            for k in range(len(tmp)):
                if cal_dis(view_encoded[i], j, tmp[k]) > 1e-8:
                    w = cal_dis(view_encoded[i], j, kth_neib_local[i][j]) / cal_dis(view_encoded[i], j, tmp[k])
                else:
                    w = 2
                tmp_weights.append(w)
            neibs_global[i].append(tmp)
            weights_global[i].append(tmp_weights)
    # print(weights_global[0][0])
    return neibs_global, neibs_local, weights_global


def get_new_knn(view_encoded, num_neibs):
    num_views = len(view_encoded)
    num_obj = view_encoded[0].shape[0]
    dim_concen = 0
    for i in range(num_views):
        dim_concen += view_encoded[i].shape[1]

    encoded_concen = np.zeros((num_obj, dim_concen))
    tmp_dim = 0
    for i in range(num_views):
        encoded_concen[:, tmp_dim: tmp_dim + view_encoded[i].shape[1]] = view_encoded[i]
        tmp_dim += view_encoded[i].shape[1]

    neibs_view = [[] for a in range(num_views)]
    for i in range(num_views):
        nbrs = NearestNeighbors(n_neighbors=num_neibs+1, algorithm='ball_tree').fit(view_encoded[i])
        distances, indices = nbrs.kneighbors(view_encoded[i])
        # print(indices)
        for j in range(num_obj):
            neibs_view[i].append((list(indices[j, 1:])))

    neibs = []
    nbrs = NearestNeighbors(n_neighbors=num_neibs+1, algorithm='ball_tree').fit(encoded_concen)
    distances, indices = nbrs.kneighbors(encoded_concen)
    # print(indices)
    for i in range(num_obj):
        neibs.append((list(indices[i, 1: ])))
    return neibs

def cal_dis(X, a, b):
    return np.sum((X[a] - X[b]) ** 2)

def cal_scores(view_encoded, num_neibs, recon_error):
    num_views = len(view_encoded)
    num_obj = view_encoded[0].shape[0]

    recon_scores = np.zeros(num_obj)
    for i in range(num_views):
        s1 = (recon_error[i] - np.min(recon_error[i])) / (np.max(recon_error[i]) - np.min(recon_error[i]))
        recon_scores += s1

    dim_concen = 0
    for i in range(num_views):
        dim_concen += view_encoded[i].shape[1]
    encoded_concen = np.zeros((num_obj, dim_concen))
    tmp_dim = 0
    for i in range(num_views):
        encoded_concen[:, tmp_dim: tmp_dim + view_encoded[i].shape[1]] = view_encoded[i]
        tmp_dim += view_encoded[i].shape[1]
    neibs = []
    NNK = NearestNeighbors(n_neighbors=num_neibs+1, algorithm='ball_tree')

    # concentrate neighbors
    nbrs = NNK.fit(encoded_concen)
    distances, indices = nbrs.kneighbors(encoded_concen)
    knn_scores = np.sum(distances, axis=1)
    tmp = list(indices[0][1:])
    list.sort(tmp)
    print('last neighbors: ')
    print(tmp)

    NNK = NearestNeighbors(n_neighbors=num_views * num_neibs + 1, algorithm='ball_tree')
    knn_scores2 = np.zeros(num_obj)
    for i in range(num_views):
        view_nbrs = NNK.fit(view_encoded[i])
        _, view_indice = view_nbrs.kneighbors(view_encoded[i])
        # print('view %d: ' % (i+1))
        # tmp = list(view_indice[0][1:])
        # list.sort(tmp)
        # print(tmp)
        for j in range(num_obj):
            for k in range(num_views):
                if i != k:
                    for l in range(1, num_neibs+1):
                        knn_scores2[j] += np.sum((view_encoded[k][j] - view_encoded[k][view_indice[j][l]]) ** 2)
    # print(np.average(knn_scores))
    return recon_scores, knn_scores

def main():
    # 打开数据集文件
    data_root_path = 'cover_data/'
    # 保存其他方法的结果
    # fo_other = open("results_other.csv", "w")
    # 保存本文方法的结果
    fo_our = open("results_our.csv", "w")

    n_objects = 101 # 指定数据集大小101 178 569 768 1484 5066 20000
    rate_outlier = [0.05, 0.05, 0.05] #一个异常率跑一个数据集，每个数据集随机生成50次
    
    auc_list = []
    for i in range(50):
        auc_list.append([])
        for dataset_name in ['zoo']:  # zoo, wine, wdbc, pima, letter_short, yeast, optdigits, cover
            for num_views in [2]:
                data_sub_path = 'data' + str(i+1) + '/5-5-5/'
                file_name = dataset_name + '-' + str(n_objects) + '-' + str(num_views) + '-' \
                            + str(rate_outlier[0]) + '-' + str(rate_outlier[1]) + '-' \
                            + str(rate_outlier[2])
                
                set_num_views(num_views)

                od_data = np.loadtxt(data_root_path + data_sub_path + file_name + '.txt', delimiter=',')
                xs = od_data[:, 0:-1] # 取除了最后一列的所有数据，存在od_data中，因为最后一列为label列
                xs = xs.astype(np.float32) # 转为float型
                labels = od_data[:, -1] # 取最后一列的数据，存在labels中
                labels = labels.astype(np.int8) # 将label转为int型，只有0，1

                set_dim(xs.shape[1]) # 获取特征维度

                print('fea num:', xs.shape[1])
                set_outlier_ratio(np.sum(labels) / xs.shape[0])

                # matplotlib.font_manager._rebuild()
                
                # 模型结果
                # mvod_main.run(data_path, dataset_name, batch_size, learning_rate, k_neibs, module_weight, albation_set)
                auc = run(data_root_path+data_sub_path, file_name, 20, 0.0001, 8, 1, (1,1))
                auc_list.append(auc)
                fo_our.write(str(round(auc,3)))
                # fo_our.write(str(round(auc,3)) + ',' + str(round(f1[2],3)) + ',')
                time.sleep(10)
                
                # 参数敏感度实验
                # for num_neibs in [4, 6, 10, 12, 14]:
                #     auc = run(data_root_path+data_sub_path, file_name, 20, 0.0001, num_neibs, 1, (1,1))
                #     auc_list[i].append(auc)
                #     fo_our.write(str(round(auc,3)) + ',')
                # time.sleep(10)
                # for module_weight in [0.25, 0.5, 2, 4]:
                #     auc = run(data_root_path+data_sub_path, file_name, 20, 0.0001, 8, 1, (1,1))
                #     fo_our.write(str(round(auc,3)) + ',')
                
                # 消融实验
                # for albation_set in [(0, 1), (1, 0)]:
                #     auc = run(data_root_path+data_sub_path, file_name, 20, 0.0001, 8, 1, albation_set)
                #     fo_our.write(str(round(auc,3)) + ',')
                # time.sleep(10)

                # fo_other.write('\n')
                # fo_other.flush()
                # fo_our.write('\n')
                fo_our.flush()
            # fo_other.write('\n')
            fo_our.write('\n')
    # fo_our.write(str(rate_outlier[0]) + '-' + str(rate_outlier[1]) + '-' \
    #                         + str(rate_outlier[2]) + dataset_name + str(num_views) + "auc average: " + str(round(np.average(auc_list),5)) + ", auc std: " + str(round(np.std(auc_list),5)) \
    #                         + ", auc max: " + str(round(np.max(auc_list),5)) + ", auc min: " + str(round(np.min(auc_list),5))) 
    fo_our.write("auc average: " + str(round(np.average(auc_list),5)) + ", auc std: " + str(round(np.std(auc_list),5)) + ", auc max: " + str(round(np.max(auc_list),5)) + ", auc min: " + str(round(np.min(auc_list),5)))       
    print("auc average: ", np.average(auc_list), ", auc std: ", np.std(auc_list), ", auc max: ", np.max(auc_list), ", auc min: ", np.min(auc_list))
    # fo_other.close()
    fo_our.close()

    return


if __name__ == '__main__':
    main()
    


