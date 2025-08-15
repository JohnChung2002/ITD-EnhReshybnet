import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import models
import utils
import data_load
import random
import ipdb
import os
import copy
import pickle

import argparse
#from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cert', choices=['cora', 'BlogCatalog', 'twitter', 'cert'], help='dataset name')
parser.add_argument('--version', type=str, default='r4.2', help='dataset version')
parser.add_argument('--data_path', type=str, default='data/', help='path to dataset')
parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
parser.add_argument('--setting', type=str, default='recon', choices=['recon', 'upsampling', 'smote', 'recon_newG', 'newG_cls', 'embed_up'], help='oversampling method')

args = parser.parse_args()

# Training setting
args_dataset = args.dataset
args_data_path = args.data_path
args_cuda = args.cuda
args_setting = args.setting
args_version = args.version

args_imbalance = True
args_opt_new_G = False
args_rec_weight = 0.000001
args_load = None
args_epochs = 10
args_up_scale = 34.93

#not used but preserved for future use
args_im_ratio = 0.5
args_model = 'sage'
args_nhid = 64
args_dropout = 0.1
args_lr = 0.001
args_weight_decay = 5e-4

'''
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args_cuda:
    torch.cuda.manual_seed(args.seed)
'''

os.makedirs('graphs', exist_ok=True)
for edge in ['none', 'device', 'email', 'user']:
    os.makedirs('graphs/{}/{}'.format(args_setting, edge), exist_ok=True)
    os.makedirs('checkpoint/{}'.format(args_dataset), exist_ok=True)
    # Load data
    if args_dataset == 'cora':
        adj, features, labels = data_load.load_data(args_data_path)
        class_sample_num = 20
        im_class_num = 3
    elif args_dataset == 'BlogCatalog':
        adj, features, labels = data_load.load_data_Blog()
        im_class_num = 14 #set it to be the number less than 100
        class_sample_num = 20 #not used
    elif args_dataset == 'twitter':
        adj, features, labels = data_load.load_sub_data_twitter()
        im_class_num = 1
        class_sample_num = 20 #not used
    elif args_dataset == 'cert':
        adj, features, labels = data_load.load_data_cert(args_data_path, args_version, edge)
        im_class_num = 1
        class_sample_num = 20 #not used
    else:
        print("no this dataset: {args_dataset}")

    print("Data loaded: ", args_dataset)

    #for artificial imbalanced setting: only the last im_class_num classes are imbalanced
    c_train_num = []
    for i in range(labels.max().item() + 1):
        if args_imbalance and i > labels.max().item()-im_class_num: #only imbalance the last classes
            c_train_num.append(int(class_sample_num*args_im_ratio))

        else:
            c_train_num.append(class_sample_num)

    #get train, validatio, test data split
    if args_dataset == 'BlogCatalog':
        idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)
    elif args_dataset == 'cora':
        idx_train, idx_val, idx_test, class_num_mat = utils.split_arti(labels, c_train_num)
    elif args_dataset == 'twitter' or args_dataset == 'cert':
        idx_train, idx_val, idx_test, class_num_mat = utils.split_genuine(labels)

    #method_1: oversampling in input domain
    if args_setting == 'upsampling':
        adj,features,labels,idx_train = utils.src_upsample(adj,features,labels,idx_train,portion=args_up_scale, im_class_num=im_class_num)

        with open("graphs/{}/{}/adj_up_{}_{}_{}.pkl".format(args_setting, edge, args_dataset, args_setting, time.time()), "wb") as f:
                pickle.dump([adj,features,labels,idx_train], f)
    if args_setting == 'smote':
        adj,features,labels,idx_train = utils.src_smote(adj,features,labels,idx_train,portion=args_up_scale, im_class_num=im_class_num)

        with open("graphs/{}/{}/adj_up_{}_{}_{}.pkl".format(args_setting, edge, args_dataset, args_setting, time.time()), "wb") as f:
            pickle.dump([adj,features,labels,idx_train], f)


    # Model and optimizer
    #if oversampling in the embedding space is required, model need to be changed
    if args_setting != 'embed_up':
        if args_model == 'sage':
            encoder = models.Sage_En(nfeat=features.shape[1],
                    nhid=args_nhid,
                    nembed=args_nhid,
                    dropout=args_dropout)
            classifier = models.Sage_Classifier(nembed=args_nhid, 
                    nhid=args_nhid, 
                    nclass=labels.max().item() + 1, 
                    dropout=args_dropout)
        elif args_model == 'gcn':
            encoder = models.GCN_En(nfeat=features.shape[1],
                    nhid=args_nhid,
                    nembed=args_nhid,
                    dropout=args_dropout)
            classifier = models.GCN_Classifier(nembed=args_nhid, 
                    nhid=args_nhid, 
                    nclass=labels.max().item() + 1, 
                    dropout=args_dropout)
        elif args_model == 'GAT':
            encoder = models.GAT_En(nfeat=features.shape[1],
                    nhid=args_nhid,
                    nembed=args_nhid,
                    dropout=args_dropout)
            classifier = models.GAT_Classifier(nembed=args_nhid, 
                    nhid=args_nhid, 
                    nclass=labels.max().item() + 1, 
                    dropout=args_dropout)
    else:
        if args_model == 'sage':
            encoder = models.Sage_En2(nfeat=features.shape[1],
                    nhid=args_nhid,
                    nembed=args_nhid,
                    dropout=args_dropout)
            classifier = models.Classifier(nembed=args_nhid, 
                    nhid=args_nhid, 
                    nclass=labels.max().item() + 1, 
                    dropout=args_dropout)
        elif args_model == 'gcn':
            encoder = models.GCN_En2(nfeat=features.shape[1],
                    nhid=args_nhid,
                    nembed=args_nhid,
                    dropout=args_dropout)
            classifier = models.Classifier(nembed=args_nhid, 
                    nhid=args_nhid, 
                    nclass=labels.max().item() + 1, 
                    dropout=args_dropout)
        elif args_model == 'GAT':
            encoder = models.GAT_En2(nfeat=features.shape[1],
                    nhid=args_nhid,
                    nembed=args_nhid,
                    dropout=args_dropout)
            classifier = models.Classifier(nembed=args_nhid, 
                    nhid=args_nhid, 
                    nclass=labels.max().item() + 1, 
                    dropout=args_dropout)



    decoder = models.Decoder(nembed=args_nhid,
            dropout=args_dropout)


    optimizer_en = optim.Adam(encoder.parameters(),
                        lr=args_lr, weight_decay=args_weight_decay)
    optimizer_cls = optim.Adam(classifier.parameters(),
                        lr=args_lr, weight_decay=args_weight_decay)
    optimizer_de = optim.Adam(decoder.parameters(),
                        lr=args_lr, weight_decay=args_weight_decay)



    if args_cuda:
        encoder = encoder.cuda()
        classifier = classifier.cuda()
        decoder = decoder.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


    def train(epoch, edge):
        t = time.time()
        encoder.train()
        classifier.train()
        decoder.train()
        optimizer_en.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_de.zero_grad()

        embed = encoder(features, adj)

        if args_setting == 'recon_newG' or args_setting == 'recon' or args_setting == 'newG_cls':
            ori_num = labels.shape[0]
            embed, labels_new, idx_train_new, adj_up = utils.recon_upsample(embed, labels, idx_train, adj=adj.detach().to_dense(),portion=args_up_scale, im_class_num=im_class_num)
            
            with open("graphs/{}/{}/adj_up_{}_{}_{}.pkl".format(args_setting, edge, args_dataset, args_setting, time.time()), "wb") as f:
                pickle.dump([embed, labels_new, idx_train_new, adj_up], f)
            
            generated_G = decoder(embed)

            loss_rec = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach().to_dense())
            
            #ipdb.set_trace()


            if not args_opt_new_G:
                adj_new = copy.deepcopy(generated_G.detach())
                threshold = 0.5
                adj_new[adj_new<threshold] = 0.0
                adj_new[adj_new>=threshold] = 1.0

                #ipdb.set_trace()
                edge_ac = adj_new[:ori_num, :ori_num].eq(adj.to_dense()).double().sum()/(ori_num**2)
            else:
                adj_new = generated_G
                edge_ac = F.l1_loss(adj_new[:ori_num, :ori_num], adj.to_dense(), reduction='mean')


            #calculate generation information
            exist_edge_prob = adj_new[:ori_num, :ori_num].mean() #edge prob for existing nodes
            generated_edge_prob = adj_new[ori_num:, :ori_num].mean() #edge prob for generated nodes
            print("edge acc: {:.4f}, exist_edge_prob: {:.4f}, generated_edge_prob: {:.4f}".format(edge_ac.item(), exist_edge_prob.item(), generated_edge_prob.item()))


            adj_new = torch.mul(adj_up, adj_new)

            exist_edge_prob = adj_new[:ori_num, :ori_num].mean() #edge prob for existing nodes
            generated_edge_prob = adj_new[ori_num:, :ori_num].mean() #edge prob for generated nodes
            print("after filtering, edge acc: {:.4f}, exist_edge_prob: {:.4f}, generated_edge_prob: {:.4f}".format(edge_ac.item(), exist_edge_prob.item(), generated_edge_prob.item()))


            adj_new[:ori_num, :][:, :ori_num] = adj.detach().to_dense()
            #adj_new = adj_new.to_sparse()
            #ipdb.set_trace()

            if not args_opt_new_G:
                adj_new = adj_new.detach()

            if args_setting == 'newG_cls':
                idx_train_new = idx_train

        elif args_setting == 'embed_up':
            #perform SMOTE in embedding space
            embed, labels_new, idx_train_new = utils.recon_upsample(embed, labels, idx_train,portion=args_up_scale, im_class_num=im_class_num)
            adj_new = adj
        else:
            labels_new = labels
            idx_train_new = idx_train
            adj_new = adj

        # #ipdb.set_trace()
        # output = classifier(embed, adj_new)



        # if args_setting == 'reweight':
        #     weight = features.new((labels.max().item()+1)).fill_(1)
        #     weight[-im_class_num:] = 1+args_up_scale
        #     loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new], weight=weight)
        # else:
        #     loss_train = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])

        # acc_train = utils.accuracy(output[idx_train], labels_new[idx_train])
        # if args_setting == 'recon_newG':
        #     loss = loss_train+loss_rec*args_rec_weight
        # elif args_setting == 'recon':
        #     loss = loss_rec + 0*loss_train
        # else:
        #     loss = loss_train
        #     loss_rec = loss_train

        # loss.backward()
        # if args_setting == 'newG_cls':
        #     optimizer_en.zero_grad()
        #     optimizer_de.zero_grad()
        # else:
        #     optimizer_en.step()

        # optimizer_cls.step()

        # if args_setting == 'recon_newG' or args_setting == 'recon':
        #     optimizer_de.step()

        # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
        # acc_val = utils.accuracy(output[idx_val], labels[idx_val])

        # #ipdb.set_trace()
        # utils.print_class_acc(output[idx_val], labels[idx_val], class_num_mat[:,1])

        # print('Epoch: {:05d}'.format(epoch+1),
        #     'loss_train: {:.4f}'.format(loss_train.item()),
        #     'loss_rec: {:.4f}'.format(loss_rec.item()),
        #     'acc_train: {:.4f}'.format(acc_train.item()),
        #     'loss_val: {:.4f}'.format(loss_val.item()),
        #     'acc_val: {:.4f}'.format(acc_val.item()),
        #     'time: {:.4f}s'.format(time.time() - t))


    def test(epoch = 0):
        encoder.eval()
        classifier.eval()
        decoder.eval()
        embed = encoder(features, adj)
        output = classifier(embed, adj)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))

        utils.print_class_acc(output[idx_test], labels[idx_test], class_num_mat[:,2], pre='test')

        '''
        if epoch==40:
            torch
        '''


    def save_model(epoch):
        saved_content = {}

        saved_content['encoder'] = encoder.state_dict()
        saved_content['decoder'] = decoder.state_dict()
        saved_content['classifier'] = classifier.state_dict()

        torch.save(saved_content, 'checkpoint/{}/{}_{}_{}_{}_{}.pth'.format(args_dataset, edge, args_setting,epoch, args_opt_new_G, args_im_ratio))

        return

    def load_model(filename):
        loaded_content = torch.load('checkpoint/{}/{}.pth'.format(args_dataset,filename), map_location=lambda storage, loc: storage)

        encoder.load_state_dict(loaded_content['encoder'])
        decoder.load_state_dict(loaded_content['decoder'])
        classifier.load_state_dict(loaded_content['classifier'])

        print("successfully loaded: "+ filename)

        return

    # Train model
    if args_load is not None:
        load_model(args_load)

    # t_total = time.time()
    for epoch in range(args_epochs):
        train(epoch, edge)

    #     if epoch % 10 == 0:
    #         test(epoch)

    #     if epoch % 100 == 0:
    #         save_model(epoch)


    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    #  test()
