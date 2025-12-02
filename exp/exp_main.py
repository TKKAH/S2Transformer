import pandas as pd
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import S2Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, simple_metric
import copy
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from utils.graph_algo import adjacency_matrix_to_dict,partition_patch, calculate_eigenvector
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        self.model_dict = {
            'S2Transformer': S2Transformer,
        }

        # 记录总开始时间
        total_start_time = time.time()
        # 1. 加载数据
        start_time = time.time()
        adj_mx = np.load('dataset/global_wind_adj.npy', allow_pickle=True)
        node_num = adj_mx.shape[0]
        load_time = time.time() - start_time
        print(f"1. 数据加载耗时: {load_time:.4f}秒")

        # 2. 图划分
        start_time = time.time()
        g = adjacency_matrix_to_dict(adj_mx)
        patch_result = partition_patch(32, 1, g, adj_mx).to(self.device)
        print(patch_result.shape)
        patch = []
        patch.append(patch_result)
        patch.append(patch_result)
        partition_time = time.time() - start_time
        print(f"2. 图划分耗时: {partition_time:.4f}秒")

        # 3. 计算子图带权邻接矩阵
        start_time = time.time()
        P,N=patch_result.shape
        adj_matrix = torch.from_numpy(adj_mx).float().to(self.device)
        intra_adj_mx = torch.zeros((P, N, N), device=adj_matrix.device)
        for p in range(P):
            node_indices = patch_result[p]
            mask = (node_indices == node_num)

            valid_indices = node_indices[~mask]  
            idx = torch.where(~mask)[0]  

            rows = idx.unsqueeze(1).expand(-1, len(idx))
            cols = idx.unsqueeze(0).expand(len(idx), -1)
            intra_adj_mx[p, rows, cols] = adj_matrix[valid_indices][:, valid_indices]
            invalid_idx = torch.where(mask)[0]  
            intra_adj_mx[p, invalid_idx, :] = 0
            intra_adj_mx[p, :, invalid_idx] = 0
        torch.diagonal(intra_adj_mx, dim1=1, dim2=2).fill_(1)
        partition_time = time.time() - start_time
        print(f"3. 计算子图内带权邻接矩阵: {partition_time:.4f}秒")

        # 4. 计算子图间带权邻接矩阵
        start_time = time.time()
        P, N = patch_result.shape
        inter_adj_mx = torch.zeros((P, P), device=adj_matrix.device)

        for i in range(P):
            for j in range(P):
                if i == j:
                    continue  

                nodes_i = patch_result[i]
                nodes_j = patch_result[j]

                nodes_i = nodes_i[nodes_i < node_num]
                nodes_j = nodes_j[nodes_j < node_num]

                sub_adj_mx = adj_matrix[nodes_i[:, None], nodes_j[None, :]]

                inter_adj_mx[i, j] = torch.max(sub_adj_mx)
    
        inter_adj_mx.diagonal().fill_(1.0)
        partition_time = time.time() - start_time
        print(f"4. 计算子图间带权邻接矩阵: {partition_time:.4f}秒")

        # 5. 计算特征
        start_time = time.time()
        node_feature = calculate_eigenvector(adj_mx, self.args.seq_len)
        eigenvector_time = time.time() - start_time
        print(f"5. 特征计算耗时: {eigenvector_time:.4f}秒")

        # 总耗时
        total_time = time.time() - total_start_time
        print(f"预处理总耗时: {total_time:.4f}秒")
        
        model = self.model_dict[self.args.model].Model(self.args,node_feature,self.device,patch,intra_adj_mx,inter_adj_mx).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            print(self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        print(f'Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/10**6} MB')
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention or self.args.model.count('Consistency') > 0:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention or self.args.model.count('Consistency') > 0:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            # torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            if self.args.pretrained_model == '':
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            else:
                self.model.load_state_dict(
                    torch.load(os.path.join('./checkpoints/' + self.args.pretrained_model, 'checkpoint.pth')))
            print('loading model finished')

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae = 0.0
        mse = 0.0
        batch_num = 0
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    batch_y = batch_y[:, -self.args.pred_len:, 0:].to(self.device)
                    input = batch_x.detach().cpu().numpy()
                    pred = outputs.detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()

                # metric
                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                if self.args.test_features == 'S_station':
                    pred = pred[:, :, self.args.target:(self.args.target + 1)]
                    true = true[:, :, self.args.target:(self.args.target + 1)]

                tmp_mae, tmp_mse = simple_metric(pred, true)
                mse += tmp_mse
                mae += tmp_mae
                batch_num += 1
                # visual
                input = batch_x.detach().cpu().numpy()
                
                gt = np.concatenate((input[0, :, 100], true[0, :, 100]), axis=0)
                pd = np.concatenate((input[0, :, 100], pred[0, :, 100]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                if i % 10 == 0:
                    print("batch: " + str(i))

        mse = mse / float(batch_num)
        mae = mae / float(batch_num)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
