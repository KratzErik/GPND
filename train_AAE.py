
# -*- coding: utf-8 -*-
# Copyright 2018 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# 
from __future__ import print_function
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from net import *
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import json
import pickle
import time
import random
import os
from utils import loadbdd100k
import datetime
from keras.preprocessing.image import load_img, img_to_array

use_cuda = torch.cuda.is_available()

FloatTensor = torch.FloatTensor
IntTensor = torch.IntTensor
LongTensor = torch.LongTensor
torch.set_default_tensor_type('torch.FloatTensor')

# If zd_merge true, will use zd discriminator that looks at entire batch.
zd_merge = False

if use_cuda:
    device = torch.cuda.current_device()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FloatTensor = torch.cuda.FloatTensor
    IntTensor = torch.cuda.IntTensor
    LongTensor = torch.cuda.LongTensor
    print("Running on ", torch.cuda.get_device_name(device))


def setup(x):
    if use_cuda:
        return x.cuda()
    else:
        return x.cpu()


def numpy2torch(x):
    return setup(torch.from_numpy(x))


def extract_batch(data, it, batch_size):
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size, :, :]) / 255.0
    #x.sub_(0.5).div_(0.5)
    return Variable(x)
None

def main(folding_id, inliner_classes, total_classes, folds=5, cfg = None):

    if cfg is None:
        print("No configuration provided, aborting...")
        return

    batch_size = cfg.batch_size
    data_train = []
    data_valid = []
    architecture = cfg.architecture
    dataset = cfg.dataset

    if dataset == "bdd100k":
        zsize = 32
        inliner_classes = [0]
        outlier_classes = [1]

        print("Data path: " + str(cfg.img_folder))
        channels = cfg.channels
        image_height = cfg.image_height
        image_width = cfg.image_width
        data_train_x, valid_imgs, _ , _ = loadbdd100k.load_bdd100k_data_filename_list(cfg.img_folder, cfg.norm_filenames, cfg.out_filenames, cfg.n_train, cfg.n_val, cfg.n_test, cfg.out_frac, image_height, image_width, channels, shuffle=cfg.shuffle)
        architecture = cfg.architecture
        experiment_name = cfg.experiment_name

        print("Transposing data to 'channels first'")
        data_train_x = np.moveaxis(data_train_x,-1,1)
        valid_imgs = np.moveaxis(valid_imgs,-1,1)

        print("Converting data from uint8 to float32")
        data_train_x = np.float32(data_train_x)
        valid_imgs = np.float32(valid_imgs)

        # Labels for training data
        data_train_y = np.zeros((len(data_train_x),),dtype=np.int)

        for img in valid_imgs:
            data_valid.append((0,img))

    elif dataset in ("dreyeve", "prosivic"):
        if cfg.architecture not in (None, "b1", "b2"):
            tmp = cfg.architecture.split("_")
            zsize = int(tmp[4])
        else:
            zsize = 512

        inliner_classes = [0]
        outlier_classes = [1]

        print("Data path: " + str(cfg.img_folder))
        architecture = cfg.architecture
        channels = cfg.channels
        image_height = cfg.image_height
        image_width = cfg.image_width
        data_train_x = [img_to_array(load_img(cfg.train_folder + filename)) for filename in os.listdir(cfg.train_folder)][:cfg.n_train]
        valid_imgs = [img_to_array(load_img(cfg.val_folder + filename)) for filename in os.listdir(cfg.val_folder)][:cfg.n_val]
        experiment_name = cfg.experiment_name

        print("Transposing data to 'channels first'")
        data_train_x = np.moveaxis(data_train_x,-1,1)
        valid_imgs = np.moveaxis(valid_imgs,-1,1)

        print("Converting data from uint8 to float32")
        data_train_x = np.float32(data_train_x)
        valid_imgs = np.float32(valid_imgs)

        # Labels for training data
        data_train_y = np.zeros((len(data_train_x),),dtype=np.int)

        for img in valid_imgs:
            data_valid.append((0,img))

    elif dataset == "mnist":
        zsize = 32
        channels = 1
        image_height = 32
        image_width = 32

        for i in range(folds):
            if i != folding_id:
                with open('data_fold_%d.pkl' % i, 'rb') as pkl:
                    fold = pickle.load(pkl, encoding='latin1')
                if len(data_valid) == 0:
                    data_valid = fold
                else:
                    data_train  += fold

        outlier_classes = []
        for i in range(total_classes):
            if i not in inliner_classes:
                outlier_classes.append(i)
        experiment_name = "-".join(str(inliner_classes))
        #keep only train classes
        data_train = [x for x in data_train if x[0] in inliner_classes]

        random.shuffle(data_train)

        def list_of_pairs_to_numpy(l):
            return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)

        data_train_x, data_train_y = list_of_pairs_to_numpy(data_train)
   
        ## End of individual dataset setups

    log_dir = cfg.log_dir
    image_dest = cfg.log_dir + 'train/'
    if not os.path.exists(image_dest):
        os.makedirs(image_dest)

    print("Train set size:", len(data_train_x))
    print("Data type:", data_train_x.dtype)
    print("Max pixel value:", np.amax(data_train_x))

    print("Configuring networks with architecture:" + architecture)
    G = Generator(zsize, channels = channels, architecture = architecture)
    setup(G)
    G.weight_init(mean=0, std=0.02)

    D = Discriminator(channels = channels, architecture = architecture)
    setup(D)
    D.weight_init(mean=0, std=0.02)

    E = Encoder(zsize, channels = channels, architecture = architecture)
    setup(E)
    E.weight_init(mean=0, std=0.02)

    if zd_merge:
        ZD = ZDiscriminator_mergebatch(zsize, batch_size).to(device)
    else:
        ZD = ZDiscriminator(zsize, batch_size).to(device)

    setup(ZD)
    ZD.weight_init(mean=0, std=0.02)

    lr = cfg.learning_rate
    betas = cfg.betas

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas = betas)
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas = betas)
    E_optimizer = optim.Adam(E.parameters(), lr=lr, betas = betas)
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas = betas)
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas = betas)

    train_epoch = cfg.n_train_epochs
    lr_change_each_ep = cfg.n_epochs_between_lr_change

    BCE_loss = nn.BCELoss()
    y_real_ = torch.ones(batch_size)
    y_fake_ = torch.zeros(batch_size)

    y_real_z = torch.ones(1 if zd_merge else batch_size)
    y_fake_z = torch.zeros(1 if zd_merge else batch_size)

    sample_size = 64
    sample = torch.randn(sample_size, zsize).view(-1, zsize, 1, 1)

    for epoch in range(train_epoch):
        G.train()
        D.train()
        E.train()
        ZD.train()

        Gtrain_loss = 0
        Dtrain_loss = 0
        Etrain_loss = 0
        GEtrain_loss = 0
        ZDtrain_loss = 0

        epoch_start_time = time.time()

        def shuffle(X):
            np.take(X, np.random.permutation(X.shape[0]), axis=0, out=X)

        shuffle(data_train_x)

        if (epoch + 1) % cfg.n_epochs_between_lr_change == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            E_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        for it in range(len(data_train_x) // batch_size):
            x = extract_batch(data_train_x, it, batch_size).view(-1, channels, image_height, image_width)
           # print("Batch type:")
           # print(type(x))
           # print("Shape of batch:")
           # print(x.shape)
            #############################################

            D.zero_grad() 

            D_result = D(x).squeeze() # removes all dimensions with size 1
            D_real_loss = BCE_loss(D_result, y_real_)

            z = torch.randn((batch_size, zsize)).view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z).detach()
 #           print("Shape of x_fake:",x_fake.shape)
            D_result = D(x_fake).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()

            D_optimizer.step()

            Dtrain_loss  += D_train_loss.item()

            #############################################

            G.zero_grad()

            z = torch.randn((batch_size, zsize)).view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z)
            D_result = D(x_fake).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            G_optimizer.step()

            Gtrain_loss  += G_train_loss.item()

            #############################################

            ZD.zero_grad()

            z = torch.randn((batch_size, zsize)).view(-1, zsize)
            z = Variable(z)

            ZD_result = ZD(z).squeeze()
            ZD_real_loss = BCE_loss(ZD_result, y_real_z)

            z = E(x).squeeze().detach()

            ZD_result = ZD(z).squeeze()
            ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

            ZD_train_loss = ZD_real_loss + ZD_fake_loss
            ZD_train_loss.backward()

            ZD_optimizer.step()

            ZDtrain_loss  += ZD_train_loss.item()

            #############################################

            E.zero_grad()
            G.zero_grad()

            z = E(x)
            x_d = G(z)

            ZD_result = ZD(z.squeeze()).squeeze()

            E_loss = BCE_loss(ZD_result, y_real_z) * 2.0

            Recon_loss = F.binary_cross_entropy(x_d, x)

            (Recon_loss + E_loss).backward()

            GE_optimizer.step()

            GEtrain_loss  += Recon_loss.item()
            Etrain_loss  += E_loss.item()

            if it == 0 and epoch % (train_epoch//10) == 0:
                comparison = torch.cat([x[:64], x_d[:64]])
                save_image(comparison.cpu(), image_dest + 'reconstruction_' + str(epoch) + '.png', nrow=64)

        Gtrain_loss /= (len(data_train_x))
        Dtrain_loss /= (len(data_train_x))
        ZDtrain_loss /= (len(data_train_x))
        GEtrain_loss /= (len(data_train_x))
        Etrain_loss /= (len(data_train_x))

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, Gloss: %.3f, Dloss: %.3f, ZDloss: %.3f, GEloss: %.3f, Eloss: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, Gtrain_loss, Dtrain_loss, ZDtrain_loss, GEtrain_loss, Etrain_loss))

        if epoch % (train_epoch//10) == 0:
            with torch.no_grad():
                resultsample = G(sample).cpu()
                save_image(resultsample.view(sample_size, channels, image_height, image_width), image_dest + 'sample_' + str(epoch) + '.png')

    model_dir = log_dir + "models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Training finish! Saving training results in ", log_dir)
    torch.save(G.state_dict(),  model_dir + "Gmodel.pkl")
    torch.save(E.state_dict(),  model_dir + "Emodel.pkl")
    torch.save(D.state_dict(),  model_dir + "Dmodel.pkl")
    torch.save(ZD.state_dict(), model_dir + "ZDmodel.pkl")

if __name__ == '__main__':
    main(0, [0], 10)

