import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import random
from utils.utils import *
from model.local_model import *
from args.args import args_parser
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from collections import Counter
from thop import profile
import time

args = args_parser()

class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha[target]
            focal_loss = alpha * focal_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss 
        
def get_one_hot(target, num_class, device):
    one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class FIFD_model:
    def __init__(self, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate, train_set, device):
        super(FIFD_model, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = network(numclass, feature_extractor)
        self.buffer_set = []
        self.numclass = 0
        self.learned_numclass = 0
        self.learned_classes = []
        self.old_model = None
        self.train_dataset = train_set
        self.signal = False
        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size
        self.train_loader = None
        self.current_class = None
        self.last_class = None
        self.task_id_old = -1
        self.device = device

    # -----------Contribution: balancing model stability and plasticity-----------
    def balance_bp_and_bs(self, features, labels):
        output = self.model(features)
        target = get_one_hot(labels, self.numclass, self.device)
        output, target = output.cuda(self.device), target.cuda(self.device)
        criterion = FocalLoss(gamma=2, alpha=None, reduction='mean')
        
        if self.old_model is None:
            loss_balance = criterion(torch.sigmoid(output), target)
            return loss_balance
        else:
            # Current task loss
            loss_balance = criterion(torch.sigmoid(output), target)
            
            # Distillation loss
            old_output = self.old_model(features)
            old_target = torch.sigmoid(old_output)
            old_task_size = old_target.shape[1]
            
            # Apply temperature scaling to both outputs
            output_scaled = output / args.temp_param
            old_output_scaled = old_output / args.temp_param
            
            # Create combined target
            distill_target = target.clone()
            distill_target[..., :old_task_size] = torch.sigmoid(old_output_scaled)
            
            # Calculate KL divergence properly
            loss_distill = F.kl_div(
                F.log_softmax(output_scaled, dim=1),
                F.softmax(distill_target, dim=1),
                reduction='batchmean'
            ) * (args.temp_param ** 2)  
            
            return args.alpha * loss_balance + (1 - args.alpha) * loss_distill
    
    # -----------Contribution: new fault detection -----------
    def new_fault_detection(self, loader, ep_g, index):
        self.model.eval()
        res = False
        if index == 0:
            class_set_attr = "class_set_0"
        elif index == 1:
            class_set_attr = "class_set_1"
        else:
            class_set_attr = "class_set_2"
        class_set = getattr(self, class_set_attr, set())
        if ep_g % args.tasks_global == 0:
            for step, (indexs, imgs, labels) in enumerate(loader):
                imgs, labels = imgs.cuda(self.device), labels.cuda(self.device)
                for label in labels:
                    if label.item() not in class_set:
                        class_set.add(label.item()) 
                        res = True
        else:
            res = False
        setattr(self, class_set_attr, class_set)
        num_class = len(class_set)
        self.weight = num_class
        self.model.train()

        return res
    
    # -----------Contribution: find Top K samples for constructing exemplar set to compensate local model-----------    
    def feature_embedding_for_exemplar_set(self, data):
        x = torch.tensor(data, dtype=torch.float32, device=self.device)
        feature_extractor_outputs = []
        batch_size = args.batch_size
        num_batches = len(x) // batch_size 
        x = x[:num_batches * batch_size]
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size]
            feature_extractor_output = self.model.feature_extractor(batch_x).detach().cpu().numpy()
            feature_extractor_outputs.append(feature_extractor_output)
        feature_extractor_output = np.concatenate(feature_extractor_outputs, axis=0)
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def train(self, model_old):
        self.model = model_to_device(self.model, False, self.device)
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.old_model = model_to_device(model_old, device=self.device, parallel=False) if model_old else None
        
        if self.old_model:
            self.old_model.eval()
            # Space complexity: Count the number of parameters (Params)
            # print(f'Loaded old model with {sum(p.numel() for p in self.old_model.parameters())} parameters')

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            all_preds = []
            all_targets = []
            label_counter = Counter()

            for step, (indexs, features, target) in enumerate(self.train_loader):
                features, target = features.to(self.device), target.to(self.device)
                label_counter.update(target.cpu().numpy().tolist())
                opt.zero_grad()
                output = self.model(features)
                # Time complexity: Compute the floating-point operations (FLOPs)  
                # flops, params = profile(self.model, inputs=(features,), verbose=False)

                if output.shape[1] == 1:  
                    preds = (torch.sigmoid(output.squeeze()) > 0.5).long()
                else:
                    preds = output.argmax(dim=1)
                
                loss = self.balance_bp_and_bs(features, target)
                loss.backward()
                opt.step()

                total_loss += loss.item() * features.size(0)  
                total_correct += (preds == target).sum().item()
                total_samples += target.size(0)

                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(target.detach().cpu().numpy())

            avg_loss = total_loss / total_samples
            accuracy = 100. * total_correct / total_samples  
            precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            # timestamp = time.strftime("%Y%m%d-%H%M%S")
            plt.rcParams['font.family'] = 'Times New Roman'

        return accuracy
    
    def train_dataloader(self, train_classes, mix):
        if mix:
            print(f'train_classes:{train_classes}')
            print(f'learned_classes:{self.learned_classes}')
            self.train_dataset.getTrainData(train_classes, self.buffer_set, self.learned_classes)
        else:
            self.train_dataset.getTrainData(train_classes, [], [])
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize,
                                  num_workers=8,
                                  pin_memory=True,
                                  drop_last=False)
        return train_loader    
    

    def beforeTrain(self, task_id_new, group):
        if task_id_new != self.task_id_old:
            self.task_id_old = task_id_new
            self.numclass = self.task_size * (task_id_new + 1)
            if group != 0:  
                if self.current_class != None:
                    self.last_class = self.current_class
                # Every client randomly samples one, two, or three novel classes from its local data for training, instead of a predefined order.
                self.current_class = random.sample([x for x in range(self.numclass - self.task_size, self.numclass)], args.task_size)
            else:
                self.last_class = None
        print(f'current_class:{self.current_class}')
        self.train_loader = self.train_dataloader(self.current_class, False)


    def update_exemplar_set(self,ep_g,index):
        self.model = model_to_device(self.model, False, self.device)
        self.model.eval()
        self.signal = False
        self.signal = self.new_fault_detection(self.train_loader,ep_g,index)

        if self.signal and (self.last_class != None):
            self.learned_numclass += len(self.last_class)
            self.learned_classes += self.last_class
            print(f'FIFD_self.learned_classes:{self.learned_classes}')
            m = int(self.memory_size / self.learned_numclass)
            self._reduce_buffer_set(m)
            for i in self.last_class: 
                features = self.train_dataset.get_features_class(i)
                print(f'features_shape:{features.shape}')
                self._construct_buffer_set(features, m)

        self.model.train()
        self.train_loader = self.train_dataloader(self.current_class, True)
                
    def _construct_buffer_set(self, features, m):
        class_mean, feature_extractor_output = self.feature_embedding_for_exemplar_set(features)
        exemplar = []
        now_class_mean = np.zeros((1, 128))
        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(features[index])
        self.buffer_set.append(exemplar)
            
    def _reduce_buffer_set(self, m):
        for index in range(len(self.buffer_set)):
            self.buffer_set[index] = self.buffer_set[index][:m]