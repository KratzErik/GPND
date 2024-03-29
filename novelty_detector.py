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
from torch.autograd.gradcheck import zero_gradients
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.stats
import os
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from utils import loadbdd100k, reuse_results
import datetime
from keras.preprocessing.image import load_img, img_to_array
from shutil import copyfile

title_size = 16
axis_title_size = 14
ticks_size = 18

power = 2.0

device = torch.device("cuda")
use_cuda = torch.cuda.is_available()

FloatTensor = torch.FloatTensor
IntTensor = torch.IntTensor
LongTensor = torch.LongTensor
torch.set_default_tensor_type('torch.FloatTensor')

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


def extract_batch(data, it, batch_size, scale = False):
    if scale:
        x = numpy2torch(data[it * batch_size:(it + 1) * batch_size]) / 255.0
    else:
        x = numpy2torch(data[it * batch_size:(it + 1) * batch_size])
    #x.sub_(0.5).div_(0.5)
    return Variable(x)

def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)


def GetF1(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2.0 * precision * recall / (precision + recall)


def main(folding_id, inliner_classes, total_classes, folds=5, cfg = None):
    # This function loads a trained AE model from cfg.log_dir/train/ and  computes the GPND 
    # novelty score for all samples in the test/in/ and test/out/ directories specified in 
    # configuration.py'. 
    # 
    # To get only novelty scores with corresponding labels, make sure 'nd_original_GPND' is 
    # set to False in 'configuration.py'. Otherwise, the original GPND program will use 
    # outlier samples to compute an "optimal threshold", and compute corresponding accuracy, 
    # F1-score, etc. 

    if cfg is None:
        print("No configuration provided, aborting...")
        return

    results_dir = cfg.log_dir + "test/"

    if not os.path.exists(results_dir):
        print("Creating directory ", results_dir)
        os.makedirs(results_dir)

    perform_tests = False
    for p in cfg.percentages:
        if cfg.test_name is None:
            wanted_result_name = results_dir + 'result_p%d.pkl'%(p)
        else:
            wanted_result_name = results_dir + 'result_%s_p%d.pkl'%(cfg.test_name,p)

        if not os.path.exists(wanted_result_name):
            print("Result not found for p = %d"%p)
            perform_tests= True
        else:
            print("Result found for p = %d"%p)

    if not perform_tests:
        print("Previous results found for all specified percentages, proceeding to evaluation")

    else:
        dataset = cfg.dataset
        zsize = 32
        batch_size = cfg.test_batch_size
        data_train = []
        data_valid = []

        def shuffle_in_unison(a, b):
            assert len(a) == len(b)
            shuffled_a = np.empty(a.shape, dtype=a.dtype)
            shuffled_b = np.empty(b.shape, dtype=b.dtype)
            permutation = np.random.permutation(len(a))
            for old_index, new_index in enumerate(permutation):
                shuffled_a[new_index] = a[old_index]
                shuffled_b[new_index] = b[old_index]
            return shuffled_a, shuffled_b

        def list_of_pairs_to_numpy(l):
                return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)

        if dataset in ("dreyeve", "prosivic"):
            inliner_classes = [0]
            outlier_classes = [1]
            print("Data path: " + str(cfg.img_folder))
            channels = cfg.channels
            image_height = cfg.image_height
            image_width = cfg.image_width

            # Load data and labels
            data_train_x = np.array([img_to_array(load_img(cfg.train_folder + filename)) for filename in os.listdir(cfg.train_folder)][:cfg.n_train])
            n_test_out = cfg.n_test - cfg.n_test_in
            data_test_x_in = np.array([img_to_array(load_img(cfg.test_in_folder + filename)) for filename in os.listdir(cfg.test_in_folder)][:cfg.n_test_in])
            data_test_x_out = np.array([img_to_array(load_img(cfg.test_out_folder + filename)) for filename in os.listdir(cfg.test_out_folder)][:n_test_out])
            print("Number of inliers: ", data_test_x_in.shape[0])
            print("Number of outliers: ", data_test_x_out.shape[0])
            data_test_x = np.concatenate([data_test_x_in, data_test_x_out])
    #        data_test_x = data_test_x_in.extend(data_test_x_out)
            test_in_labels = np.zeros((cfg.n_test_in,),dtype=np.int32)
            test_out_labels = np.ones((n_test_out,),dtype=np.int32)
            test_labels = np.concatenate([test_in_labels, test_out_labels])

            architecture = cfg.architecture
            experiment_name = cfg.experiment_name
            if cfg not in ("b1", "b2"):
                tmp = cfg.architecture.split("_")
                zsize = int(tmp[4])

            print("Transposing data to 'channels first'")
            data_train_x = np.moveaxis(data_train_x,-1,1)
            data_test_x = np.moveaxis(data_test_x,-1,1)

            print("Converting data from uint8 to float32")
            data_train_x = np.float32(data_train_x)
            data_test_x = np.float32(data_test_x)

            # Labels for training data
            data_train_y = np.zeros((len(data_train_x),),dtype=np.int)

            if cfg.nd_original_GPND:
                # Test and validation data both have outliers: split in two parts
                data_valid = [(lbl,img) for i, lbl, img in zip(range(len(test_labels)),test_labels, data_test_x) if i % 2 is 0]
                data_test = [(lbl,img) for i, lbl, img in zip(range(len(test_labels)),test_labels, data_test_x) if i % 2 is not 0]
            else: # SMILE project: all test_data is for test, validation data not used
                data_test = [(lbl,img) for lbl, img in zip(test_labels, data_test_x)]

        if dataset == "bdd100k":
            inliner_classes = [0]
            outlier_classes = [1]

            print("Data path: " + str(cfg.img_folder))
            channels = cfg.channels
            image_height = cfg.image_height
            image_width = cfg.image_width
            data_train_x, _, data_test_x , test_labels = loadbdd100k.load_bdd100k_data_filename_list(cfg.img_folder, cfg.norm_filenames, cfg.out_filenames, cfg.n_train, cfg.n_val, cfg.n_test, cfg.out_frac, image_height, image_width, channels, shuffle=cfg.shuffle)
            architecture = cfg.architecture
            experiment_name = cfg.experiment_name

            print("Transposing data to 'channels first'")
            data_train_x = np.moveaxis(data_train_x,-1,1)
            data_test_x = np.moveaxis(data_test_x,-1,1)

            print("Converting data from uint8 to float32")
            data_train_x = np.float32(data_train_x)
            data_test_x = np.float32(data_test_x)

            # Labels for training data
            data_train_y = np.zeros((len(data_train_x),),dtype=np.int)

            # Test and validation data both have outliers: split in two parts
            data_valid = [(lbl,img) for i, lbl, img in zip(range(len(test_labels)),test_labels, data_test_x) if i % 2 is 0]
            data_test = [(lbl,img) for i, lbl, img in zip(range(len(test_labels)),test_labels, data_test_x) if i % 2 is not 0]

        elif dataset == "mnist":
            outlier_classes = []
            architecture = None
            image_height = 32
            image_width = 32
            channels = 1
            experiment_name = "-".join(str(inliner_classes))

            for i in range(total_classes):
                if i not in inliner_classes:
                    outlier_classes.append(i)

            for i in range(folds):
                if i != folding_id:
                    with open('data_fold_%d.pkl' % i, 'rb') as pkl:
                        fold = pickle.load(pkl, encoding='latin1')
                    if len(data_valid) == 0:
                        data_valid = fold
                    else:
                        data_train += fold

            with open('data_fold_%d.pkl' % folding_id, 'rb') as pkl:
                data_test = pickle.load(pkl, encoding='latin1')

            #keep only train classes
            data_train = [x for x in data_train if x[0] in inliner_classes]

            random.seed(0)
            random.shuffle(data_train)

            data_train_x, data_train_y = list_of_pairs_to_numpy(data_train)

        # Rescale before training (not batchwise)
        data_train_x /= 255.0

        print("Train set size:", len(data_train_x))

        G = Generator(zsize, channels = channels, architecture = architecture).to(device)
        E = Encoder(zsize, channels = channels, architecture = architecture).to(device)
        setup(E)
        setup(G)
        G.eval()
        E.eval()

        G.load_state_dict(torch.load(cfg.log_dir + "models/Gmodel.pkl"))
        E.load_state_dict(torch.load(cfg.log_dir + "models/Emodel.pkl"))

        sample_size = 64
        sample = torch.randn(sample_size, zsize).to(device)
        sample = G(sample.view(-1, zsize)).cpu()
        save_image(sample.view(sample_size, channels, image_height, image_width), results_dir +  'Generator_sample.png')

        if True:
            zlist = []
            rlist = []

            for it in range(len(data_train_x) // batch_size):
                x = Variable(extract_batch(data_train_x, it, batch_size).view(-1, channels * image_height * image_width).data, requires_grad=True)
                z = E(x.view(-1, channels, image_height, image_width))
                recon_batch = G(z)
                z = z.squeeze()

                recon_batch = recon_batch.squeeze().cpu().detach().numpy()
                x = x.squeeze().cpu().detach().numpy()

                z = z.cpu().detach().numpy()

                for i in range(batch_size):
                    distance = np.sum(np.power(recon_batch[i].flatten() - x[i].flatten(), power))
                    rlist.append(distance)

                zlist.append(z)

            data = {}
            data['rlist'] = rlist
            data['zlist'] = zlist

            with open('data.pkl', 'wb') as pkl:
                pickle.dump(data, pkl)

        with open('data.pkl', 'rb') as pkl:
            data = pickle.load(pkl)

        rlist = data['rlist']
        zlist = data['zlist']

        counts, bin_edges = np.histogram(rlist, bins=30, normed=True)

        plt.plot(bin_edges[1:], counts, linewidth=2)
        plt.xlabel(r"Distance, $\left \|\| I - \hat{I} \right \|\|$", fontsize=axis_title_size)
        plt.ylabel('Probability density', fontsize=axis_title_size)
        plt.title(r"PDF of distance for reconstruction error, $p\left(\left \|\| I - \hat{I} \right \|\| \right)$", fontsize=title_size)
        plt.grid(True)
        plt.xticks(fontsize=ticks_size)
        plt.yticks(fontsize=ticks_size)
        plt.tight_layout(rect=(0.0, 0.0, 1, 0.95))
        str_tmp = results_dir + 'data_d%d_randomsearch.pdf'
        plt.savefig(str_tmp % inliner_classes[0])
        str_tmp = results_dir + 'data_d%d_randomsearch.eps'
        plt.savefig(str_tmp % inliner_classes[0])
        plt.clf()
        plt.cla()
        plt.close()

        def r_pdf(x, bins, count):
            if x < bins[0]:
                return max(count[0], 1e-308)
            if x >= bins[-1]:
                return max(count[-1], 1e-308)
            id = np.digitize(x, bins) - 1
            return max(count[id], 1e-308)

        zlist = np.concatenate(zlist)
    #    for i in range(zsize):
    #        plt.hist(zlist[:, i], bins='auto', histtype='step')

        plt.xlabel(r"$z$", fontsize=axis_title_size)
        plt.ylabel('Probability density', fontsize=axis_title_size)
        plt.title(r"PDF of embeding $p\left(z \right)$", fontsize=title_size)
        plt.grid(True)
        plt.xticks(fontsize=ticks_size)
        plt.yticks(fontsize=ticks_size)
        plt.tight_layout(rect=(0.0, 0.0, 1, 0.95))
        str_tmp = results_dir + 'data_d%d_embeding.pdf'
        plt.savefig(str_tmp  % inliner_classes[0])
        str_tmp = results_dir + 'data_d%d_embeding.eps'
        plt.savefig(str_tmp  % inliner_classes[0])
        plt.clf()
        plt.cla()
        plt.close()

        gennorm_param = np.zeros([3, zsize])
        for i in range(zsize):
            betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i])
            gennorm_param[0, i] = betta
            gennorm_param[1, i] = loc
            gennorm_param[2, i] = scale

        def compute_threshold(data_valid, percentage):
            #############################################################################################
            # Searching for threshold on validation set
            random.shuffle(data_valid)
            data_valid_outlier = [x for x in data_valid if x[0] in outlier_classes]
            data_valid_inliner = [x for x in data_valid if x[0] in inliner_classes]

            inliner_count = len(data_valid_inliner)
            outlier_count = inliner_count * percentage // (100 - percentage)

            if len(data_valid_outlier) > outlier_count:
                data_valid_outlier = data_valid_outlier[:outlier_count]
            else:
                outlier_count = len(data_valid_outlier)
                inliner_count = outlier_count * (100 - percentage) // percentage
                data_valid_inliner = data_valid_inliner[:inliner_count]

            _data_valid = data_valid_outlier + data_valid_inliner
            random.shuffle(_data_valid)

            data_valid_x, data_valid_y = list_of_pairs_to_numpy(_data_valid)
            result = []
            novel = []

            for it in range(len(data_valid_x) // batch_size):
                x = Variable(extract_batch(data_valid_x, it, batch_size).view(-1, channels * image_height * image_width).data, requires_grad=True)
                label = extract_batch(data_valid_y, it, batch_size)

                z = E(x.view(-1, channels, image_height, image_width))
                recon_batch = G(z)
                z = z.squeeze()

                J = compute_jacobian(x, z)
                J = J.cpu().numpy()
                z = z.cpu().detach().numpy()

                recon_batch = recon_batch.squeeze().cpu().detach().numpy()
                x = x.squeeze().cpu().detach().numpy()

                for i in range(batch_size):
                    # The svd step accounts for >99.9% of the time when processing large images (tested for 256x256pixels)
                    u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                    logD = np.sum(np.log(np.abs(s))) # | \mathrm{det} S^{-1} |

                    p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                    logPz = np.sum(np.log(p))

                    # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                    # In this case, just assign some large negative value to make sure that the sample 
                    # is classified as unknown. Edit by Kratz: same with logD. 
                    if not np.isfinite(logPz):
                        logPz = -1000

                    if not np.isfinite(logD):
                        logD = -1000


                    distance = np.sum(np.power(x[i].flatten() - recon_batch[i].flatten(), power))

                    logPe = np.log(r_pdf(distance, bin_edges, counts)) # p_{\|W^{\perp}\|} (\|w^{\perp}\|)
                    logPe -= np.log(distance) * (channels*image_height * image_width - zsize) # \| w^{\perp} \|}^{m-n}

                    P = logD + logPz + logPe

                    result.append(P)
                    novel.append(label[i].item() in inliner_classes)

            result = np.asarray(result, dtype=np.float32)
            novel = np.asarray(novel, dtype=np.float32)

            minP = min(result) - 1
            maxP = max(result) + 1
            #print(maxP)

            best_e = 0
            best_f = 0
            best_e_ = 0
            best_f_ = 0

            not_novel = np.logical_not(novel)

            max_vals = 10000
            print("maxP,minP:",maxP, minP)
            if (maxP-minP)//0.1 > max_vals:
                p_range = np.linspace(minP,maxP,num=max_vals)
            else:
                p_range = np.arange(minP, maxP, 0.1)
            for e in p_range:
                y = np.greater(result, e)

                true_positive = np.sum(np.logical_and(y, novel))
                false_positive = np.sum(np.logical_and(y, not_novel))
                false_negative = np.sum(np.logical_and(np.logical_not(y), novel))

                if true_positive > 0:
                    f = GetF1(true_positive, false_positive, false_negative)
                    if f > best_f:
                        best_f = f
                        best_e = e
                    if f >= best_f_:
                        best_f_ = f
                        best_e_ = e

            best_e = (best_e + best_e_) / 2.0

            print("Best e: ", best_e)
            return best_e

    def test(data_test, percentage, e = None):

        # Set recon_loss mode from config file
        if cfg.loss.lower() == "bce":
            recon_loss = nn.BCELoss()
        elif cfg.loss.lower() == "l2":
            recon_loss = nn.MSELoss()
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        random.shuffle(data_test)
        data_test_outlier = [x for x in data_test if x[0] in outlier_classes]
        data_test_inliner = [x for x in data_test if x[0] in inliner_classes]

        inliner_count = len(data_test_inliner)
        outlier_count = inliner_count * percentage // (100 - percentage)

        if len(data_test_outlier) >= outlier_count:
            print("Found enough outliers for correct percentage, using specified inlier count")
            data_test_outlier = data_test_outlier[:outlier_count]
        else:
            print("Not enough outliers for correct percentage, adjusting:\ninliers: %d, outliers:%d -> %d%"%(inliner_count, outlier_count,len(data_test_outlier)))
            outlier_count = len(data_test_outlier)
            inliner_count = outlier_count * (100 - percentage) // percentage
            data_test_inliner = data_test_inliner[:inliner_count]
        print("Inliers: %d\nOutliers: %d"%(len(data_test_inliner), len(data_test_outlier)))
        data_test = data_test_outlier + data_test_inliner
        random.shuffle(data_test)

        data_test_x, data_test_y = list_of_pairs_to_numpy(data_test)

        # Rescale test data
        data_test_x /= 255.0
        count = 0

        recon_losses = []
        result = []
        n_batches = len(data_test_x)//batch_size
        print("Testing %d batches of size %d"%(n_batches, batch_size))
        total_time = 0
        for it in range(n_batches):
            start_time = time.time()

            x = Variable(extract_batch(data_test_x, it, batch_size).view(-1, channels * image_height * image_width).data, requires_grad=True)
            label = extract_batch(data_test_y, it, batch_size)

            z = E(x.view(-1, channels, image_height, image_width))
            recon_batch = G(z)

            # Compute reconstruction error for comparison with GPND result
#            recon_losses_batch = recon_loss(x,Variable(recon_batch))
#            recon_losses_batch = recon_losses_batch.cpu().detach().numpy()
#            recon_losses_batch = np.sum(recon_losses_batch,axis=1)
#            recon_losses.extend(recon_losses_batch)


            z = z.squeeze()
            jac_start_time = time.time()
            J = compute_jacobian(x, z)
            jac_end_time = time.time()
            print("Time to compute jacobian:",jac_end_time-jac_start_time)
            J = J.cpu().numpy()

            z = z.cpu().detach().numpy()

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            for i in range(batch_size):
                im_start = time.time()
                u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                svd_end = time.time()
                print("SVD decomp. time: ",svd_end-im_start)
                logD = np.sum(np.log(np.abs(s)))

                p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                logPz = np.sum(np.log(p))

                # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                # In this case, just assign some large negative value to make sure that the sample 
                # is classified as unknown. 
                if not np.isfinite(logPz):
                    logPz = -1000

                if not np.isfinite(logD):
                    logD = -1000

                distance = np.sum(np.power(x[i].flatten() - recon_batch[i].flatten(), power))

                logPe = np.log(r_pdf(distance, bin_edges, counts))
                logPe -= np.log(distance) * (channels * image_height * image_width - zsize)
                
                count += 1

                P = logD + logPz + logPe
                
                if e is not None:
                    if (label[i].item() in inliner_classes) != (P > e):
                        if not label[i].item() in inliner_classes:
                            false_positive += 1
                        if label[i].item() in inliner_classes:
                            false_negative += 1
                    else:
                        if label[i].item() in inliner_classes:
                            true_positive += 1
                        else:
                            true_negative += 1

                result.append(((label[i].item() in outlier_classes), -P))
                print("Image time: ",time.time()-im_start)
                print("Batch %d/%d: image %d/%d"%(it+1,len(data_test_x)//batch_size,i+1, batch_size))
            per_batch_time = time.time()-start_time
            total_time += per_batch_time 
            complete_batches = it + 1
            remaining_time = (n_batches - complete_batches)*total_time/complete_batches
            print("Batch %d/%d complete\ttime: %.2f\tETA:%dh%dm%.1fs"%(it+1,len(data_test_x)//batch_size, per_batch_time, remaining_time//3600, (remaining_time%3600)//60, remaining_time%60))

        error = 100 * (true_positive + true_negative) / count

        y_true = [x[0] for x in result]
        y_scores = [x[1] for x in result]

        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0

        # Pickle results, they are then extracted by other function to produce metrics
        if cfg.test_name is None:
            results_path = results_dir + 'result_p%d.pkl'%(percentage)
        else:
            results_path = results_dir + 'result_%s_p%d.pkl'%(cfg.test_name,percentage)
        with open(results_path, 'wb') as output:
            pickle.dump([result,recon_losses], output)
    
        if cfg.nd_original_GPND:

            print("Percentage ", percentage)
            print("Error ", error)
            f1 = GetF1(true_positive, false_positive, false_negative)
            print("F1 ", GetF1(true_positive, false_positive, false_negative))
            print("AUC ", auc)

            #inliers
            X1 = [x[1] for x in result if not x[0]]

            #outliers
            Y1 = [x[1] for x in result if x[0]]

            minP = min([x[1] for x in result]) - 1
            maxP = max([x[1] for x in result]) + 1


            # For looping over values of e:
            max_vals = 10000
            if (maxP-minP)//0.2 > max_vals:
                p_range = np.linspace(minP,maxP,num=max_vals)
            else:
                p_range = np.arange(minP, maxP, 0.2)
            ##################################################################
            # FPR at TPR 95
            ##################################################################
            fpr95 = 0.0
            clothest_tpr = 1.0
            dist_tpr = 1.0
            for e in p_range:
                tpr = np.sum(np.greater_equal(X1, e)) / np.float(len(X1))
                fpr = np.sum(np.greater_equal(Y1, e)) / np.float(len(Y1))
                if abs(tpr - 0.95) < dist_tpr:
                    dist_tpr = abs(tpr - 0.95)
                    clothest_tpr = tpr
                    fpr95 = fpr

            print("tpr: ", clothest_tpr)
            print("fpr95: ", fpr95)

            ##################################################################
            # Detection error
            ##################################################################
            error = 1.0
            for e in p_range:
                tpr = np.sum(np.less(X1, e)) / np.float(len(X1))
                fpr = np.sum(np.greater_equal(Y1, e)) / np.float(len(Y1))
                error = np.minimum(error, (tpr + fpr) / 2.0)

            print("Detection error: ", error)

            ##################################################################
            # AUPR IN
            ##################################################################
            auprin = 0.0
            recallTemp = 1.0
            for e in p_range:
                tp = np.sum(np.greater_equal(X1, e))
                fp = np.sum(np.greater_equal(Y1, e))
                if tp + fp == 0:
                    continue
                precision = tp / (tp + fp)
                recall = tp / np.float(len(X1))
                auprin += (recallTemp-recall)*precision
                recallTemp = recall
            auprin += recall * precision

            print("auprin: ", auprin)


            ##################################################################
            # AUPR OUT
            ##################################################################
            minp, maxP = -maxP, -minP

            if (maxP-minP)//0.2 > max_vals:
                p_range = np.linspace(minP,maxP,num=max_vals)
            else:
                p_range = np.arange(minP, maxP, 0.2)

            X1 = [-x for x in X1]
            Y1 = [-x for x in Y1]
            auprout = 0.0
            recallTemp = 1.0
            for e in p_range:
                tp = np.sum(np.greater_equal(Y1, e))
                fp = np.sum(np.greater_equal(X1, e))
                if tp + fp == 0:
                    continue
                precision = tp / (tp + fp)
                recall = tp / np.float(len(Y1))
                auprout += (recallTemp-recall)*precision
                recallTemp = recall
            auprout += recall * precision

            print("auprout: ", auprout)

            with open(results_dir + "results_%s.txt", "a") as file:
                file.write(
                    "Class: %d\n Percentage: %d\n"
                    "Error: %f\n F1: %f\n AUC: %f\nfpr95: %f"
                    "\nDetection: %f\nauprin: %f\nauprout: %f\n\n" %
                    (inliner_classes[0], percentage, error, f1, auc, fpr95, error, auprin, auprout))

            return auc, f1, fpr95, error, auprin, auprout

        else:
            return result, total_time

    if cfg.nd_original_GPND and perform_tests: # This is used in original GPND experiment
        percentages = cfg.percentages

        results = {}

        for p in percentages:
            e = compute_threshold(data_valid, p)
            results[p] = test(data_test, p, e)

        return results

    else: # this is used in SMILE experiments, to get only metrics specified in reuse_results.get_performace_metrics
        log = ["Testing started at %s"%time.strftime("%a, %d %b %Y %H:%M:%S UTC")]
        total_time = []

        for p in cfg.percentages:
            if (cfg.test_name is None and os.path.exists(results_dir + 'result_p%d.pkl'%(p))) or (cfg.test_name is not None and os.path.exists(results_dir + 'result_%s_p%d.pkl'%(cfg.test_name,p))) :
                total_time.append(0)
                print("Result already found: p = %d"%p)
            else:
                _, percentage_time = test(data_test,p)
                total_time.append(percentage_time)

        log.append("Results for experiment:")
        log.append("Inliers: %s"%cfg.outliers_name)
        log.append("Outliers: %s"%cfg.inliers_name)

        metrics_str = reuse_results.get_performance_metrics() # loads pickled results for experiment and percentages specified in cfg
        for i, line in enumerate(metrics_str):
            line += "\tTest time for percentage:\t%dh%dm%.1fs"%(total_time[i]//3600, (total_time[i]%3600)//60, total_time[i]%60)
            log.append(line)

        print("Logging test configuration and results:")
        with open(cfg.log_dir + "configuration.py",'a') as f_out:
            for line in log:
                print(line)
                f_out.write(line+'\n')

        # Export pickled result to common results repo
        if len(cfg.percentages) == 1 and cfg.export_results:
            reuse_results.export_scores()
        print('\n\n')
if __name__ == '__main__':
    main(0, [0], 10)
