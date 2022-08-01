import os
import sys
import argparse
import traceback
import copy
import math
import random
import pickle
import csv
import gc
import IPython
from itertools import combinations

import numpy as np
import torch.nn as nn
import hparams as hp
import torch
import seaborn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from scipy.spatial.distance import cosine

from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.utils as vutils
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
import tqdm
import util
import voca
from voca import WordVocab

#sys.path.extend(["../","./"])

# Code largely borrowed from
# https://github.com/huanghonggit/Mask-Language-Model

class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)

class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """
    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=hp.self_att_dropout_rate)
        self.attention = None

    def forward(self, key, value, query, mask=None, query_mask=None):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        # Masking to ignore padding (key side)
        if mask is not None:

            attn = attn.masked_fill(mask, -2 ** 32 + 1)
            attn = torch.softmax(attn, dim=-1)
        else:
            attn = torch.softmax(attn, dim=-1)

        # Masking to ignore padding (query side)
        if query_mask is not None:
            attn = attn * query_mask  # 128,804,64

        # Dropout
        # attn = self.attn_dropout(attn)
        self.attention = self.attn_dropout(attn).view(key.size(0)//4, 4, -1, key.size(1))

        # Get Context Vector
        result = torch.bmm(attn, value)

        return result, self.attention  # 128,804,158

class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self):
        """
        K: key, V: value, Q: query
        """
        super(Attention, self).__init__()

        self.num_hidden = hp.num_hidden
        self.num_attn_heads = hp.num_attn_heads
        self.num_hidden_per_attn = self.num_hidden // self.num_attn_heads

        self.K = Linear(self.num_hidden, self.num_hidden, bias=False)
        self.V = Linear(self.num_hidden, self.num_hidden, bias=False)
        self.Q = Linear(self.num_hidden, self.num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=hp.self_att_block_res_dropout)
        self.final_linear = Linear(self.num_hidden * 2, self.num_hidden)
        self.layer_norm = nn.LayerNorm(self.num_hidden)

    def forward(self, memory, decoder_input, mask=None, query_mask=None):

        batch_size = memory.size(0)
        seq_k = memory.size(1)
        seq_q = decoder_input.size(1)

        # Repeat masks num_attn_heads times
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)
            # (32,169,169) -> (128,169,169)
            query_mask = query_mask.repeat(self.num_attn_heads, 1, 1)
        if mask is not None:
            # (32,169,169) -> (128,169,169)
            mask = mask.repeat(self.num_attn_heads, 1, 1)

        # Generate a multihead attention
        K = self.K(memory).view(batch_size, seq_k, self.num_attn_heads,
                                self.num_hidden_per_attn)
        V = self.V(memory).view(batch_size, seq_k, self.num_attn_heads,
                                self.num_hidden_per_attn)
        Q = self.Q(decoder_input).view(batch_size, seq_q, self.num_attn_heads,
                                       self.num_hidden_per_attn)

        K = K.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        V = V.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        Q = Q.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get a context vector
        result, attns = self.multihead(K, V, Q, mask=mask, query_mask=query_mask)

        # Concatenate all multihead context vectors
        result = result.view(self.num_attn_heads, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        # Concatenate a context vector with a single input (most important)
        #   e.g. input (256) + result (256) -> 512
        result = torch.cat([decoder_input, result], dim=-1)

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + decoder_input

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns

class Convolution(nn.Module):
    """
    Convolution Module
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Convolution, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Neural Network (FFNN)
    """

    def __init__(self):
        super(PositionWiseFeedForward, self).__init__()

        num_hidden = hp.num_hidden
        self.w_1 = Convolution(num_hidden, num_hidden * 4, kernel_size=1,
                               w_init='relu')
        self.w_2 = Convolution(num_hidden * 4, num_hidden, kernel_size=1,
                               w_init='linear')
        self.dropout = nn.Dropout(p=hp.pos_dropout_rate)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, input_):
        # FFNN Network
        x = input_.transpose(1, 2)
        x = self.w_2(torch.relu(self.w_1(x)))
        x = x.transpose(1, 2)

        # residual connection
        x = x + input_

        # dropout
        x = self.dropout(x)

        # layer normalization
        x = self.layer_norm(x)

        return x

class EncoderPrenet(nn.Module):
    """
    Pre-network for Encoder consists of convolution networks.
    """
    def __init__(self, embedding_size, channels):
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size

        self.conv1d_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        in_channels, out_channels = embedding_size, channels
        kernel_size = hp.enc_conv1d_kernel_size
        for i in range(hp.enc_conv1d_layers):
            conv1d = Convolution(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          padding=int(np.floor(kernel_size / 2)), w_init='relu')
            self.conv1d_layers.append(conv1d)

            batch_norm = nn.BatchNorm1d(out_channels)
            self.bn_layers.append(batch_norm)

            dropout_layer = nn.Dropout(p=hp.enc_conv1d_dropout_rate)
            self.dropout_layers.append(dropout_layer)

            in_channels, out_channels = out_channels, out_channels

        self.projection = Linear(out_channels, channels)

    def forward(self, input):
        """
        :param input: Batch * Token * dimension
        :return:
        """
        input = input.transpose(1, 2)       # B*d*T

        for conv1d, bn, dropout in zip(self.conv1d_layers, self.bn_layers, self.dropout_layers):
            input = dropout(torch.relu(bn(conv1d(input))))     # B*d*T

        input = input.transpose(1, 2)       # B*T*d
        input = self.projection(input)

        return input

class BERTEncoder(nn.Module):
    """
    Encoder Network
    """
    def __init__(self, args=None):
        """
        :param embedding_size: dimension of embedding
        :param num_hidden: dimension of hidden
        """
        super(BERTEncoder, self).__init__()
        self.num_hidden = hp.num_hidden
        self.embed_dim = hp.embed_dim
        self.enc_maxlen = hp.enc_maxlen
        self.num_attn_layers = hp.num_attn_layers
        self.num_attn_heads = hp.num_attn_heads
        self.num_hidden = hp.num_hidden
        self.vocab_size = args.vocab_size

        self.alpha = nn.Parameter(torch.ones(1))
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(
            self.get_sinusoid_encoding_table(self.enc_maxlen, self.num_hidden,
                                        padding_idx=0), freeze=True)
        self.pos_dropout = nn.Dropout(p=hp.pos_dropout_rate)
        self.encoder_prenet = EncoderPrenet(self.embed_dim, self.num_hidden)
        self.layers = self.clones(Attention(), self.num_attn_layers)
        self.pwffns = self.clones(PositionWiseFeedForward(), self.num_attn_layers)

        # self.init_model()
        # self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)

    def forward(self, x, pos):
        if self.training:
            c_mask = x.ne(0).type(torch.float)
            mask = x.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)

        else:
            c_mask, mask = None, None

        x = self.embed(x)           # B*T*d
        x = self.encoder_prenet(x)  # Three convolutions

        # Get a positional embedding with an alpha and add them
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x

        # Positional dropout
        x = self.pos_dropout(x)

        # Attention encoder-encoder
        attns = list()
        xs = list()
        for layer, pwffn in zip(self.layers, self.pwffns):
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)
            x = pwffn(x)
            xs.append(x)
            attns.append(attn)

        return x, attns, xs

    def checkpoint(self, path, steps):
        self.save(f'{path}/mlm_checkpoint_{steps}steps.pyt')

    def save(self, path):
        torch.save(self.state_dict(), path)

    def clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def get_sinusoid_encoding_table(self, n_position, d_hid, padding_idx=None):
        ''' Sinusoid position encoding table '''

        def cal_angle(position, hid_idx):  # 0-255
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):  # 0-1023
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)]) # 1024，256  1-1024  1024

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.

        return torch.FloatTensor(sinusoid_table)


class BERTLanguageModel(nn.Module):
    """
    BERT Language Model
    Masked Language Model
    """

    def __init__(self, bert: BERTEncoder, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.num_hidden, vocab_size)
        self.init_model()

    def forward(self, x, pos):
        x, attn_list, xs = self.bert(x, pos)
        return self.mask_lm(x), attn_list

    def init_model(self):
        un_init = ['bert.embed.weight', 'bert.pos_emb.weight']
        for n, p in self.named_parameters():
            if n not in un_init and p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}

class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=hp.adam_beta1, b2=hp.adam_beta2, e=hp.epsilon,
                 weight_decay_rate=hp.adam_weight_decay_rate,
                 max_grad_norm=1.0):
        assert lr > 0.0, "Learning rate: %f - should be > 0.0" % (lr)
        assert schedule in SCHEDULES, "Invalid schedule : %s" % (schedule)
        assert 0.0 <= warmup < 1.0 or warmup == -1.0, \
            "Warmup %f - should be in 0.0 ~ 1.0 or -1 (no warm up)" % (warmup)
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        """ get learning rate in training """
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if not state:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, '
                                       'please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if not state:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss

def optim4GPU(model, total_steps):
    """ optimizer for GPU training """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}]
    return BertAdam(optimizer_grouped_parameters,
                    lr=hp.lr,
                    warmup=hp.warmup,
                    t_total=total_steps)

class BERTTrainer():
    """
    BERTTrainer make the pretrained BERT model with two LM training method.
        1. Masked Language Model : 3.3.1 Task #1: Masked LM
    please check the details on README.md with simple example.
    """

    def __init__(self, bert: BERTEncoder, vocab_size: int, cuda,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader = None,
                 log_freq: int = hp.log_freq, global_step=0, path=None):
        """
        :param bert: MLM model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param valid_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param log_freq: logging frequency of the batch iteration
        """

        self.step = global_step
        self.path = path

        # Setup a cuda device for BERT training
        has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:"+cuda if has_cuda else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert

        # Initialize a BERT Language Model with a BERT model
        self.model = BERTLanguageModel(bert, vocab_size).to(self.device)

        '''
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        '''

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader

        # Setting the Adam optimizer with hyper-param
        total_steps = hp.epochs * len(self.train_data)
        self.optimer = optim4GPU(self.model, total_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq
        self.num_params()

    def train(self):
        try:
            for epoch in range(hp.epochs):
                # Setting a tqdm progress bar
                data_iter = tqdm.tqdm(enumerate(self.train_data),
                                      desc="[+] EP_%s (%d)" % ("train", epoch),
                                      total=len(self.train_data),
                                      bar_format="{l_bar}{r_bar}")

                running_loss = 0
                for i, data in data_iter:
                    self.step += 1

                    # 0. batch_data will be sent into the device (GPU or cpu)
                    data = {key: value.to(self.device) for key, value in data.items()}

                    # 1. forward masked_lm model
                    mask_lm_output, attn_list = \
                        self.model.forward(data["mlm_input"], data["input_position"])

                    # 2. NLLLoss of predicting masked token word
                    self.optimer.zero_grad()
                    loss = self.criterion(mask_lm_output.transpose(1, 2), data["mlm_label"])

                    # 3. backward and optimization only in train
                    loss.backward()
                    self.optimer.step()

                    # loss
                    running_loss += loss.item()
                    avg_loss = running_loss / (i + 1)

                    # write a log
                    post_fix = "\tEpoch:%2d, Iter:%5d, Step:%5d, AvgLoss: %.6f, Loss: %.6f" \
                              % (epoch, i, self.step, avg_loss, loss.item())

                    if i % self.log_freq == 0:
                        data_iter.write(str(post_fix))

                    # save model checkpoint
                    #if self.step % hp.save_checkpoint == 0:
                    #    self.bert.checkpoint(self.path.bert_checkpoints_path, self.step)

                    # save the bert model
                    if self.step % hp.save_model == 0:
                        self.save_bert_model(epoch, f"{self.path.bert_path}/bert")

                gc.collect()
                torch.cuda.empty_cache()
                # Evaluate the model after each epoch
                valid_loss = self.evaluate(epoch)

                # Save the model after each epoch
                self.save_bert_model(epoch, f"{self.path.bert_path}/bert")
                self.save_mlm_model(epoch, f"{self.path.mlm_path}/mlm")
                print(f"EP_{epoch}, train_avg_loss={avg_loss}, valid_avg_loss={valid_loss}")

        except BaseException:
            traceback.print_exc()

    def evaluate(self, epoch):
        self.model.eval()

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.valid_data),
                              desc="[+] EP_%s (%d)" % ("valid", epoch),
                              total=len(self.valid_data),
                              bar_format="{l_bar}{r_bar}")

        running_loss = 0
        with torch.no_grad():
            for i, data in data_iter:
                self.step += 1

                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward masked_lm model
                mask_lm_output, attn_list = \
                    self.model.forward(data["mlm_input"], data["input_position"])

                # 2. NLLLoss of predicting masked token word
                loss = self.criterion(mask_lm_output.transpose(1, 2), data["mlm_label"])

                # loss
                running_loss += loss.cpu().detach().numpy()
                avg_loss = running_loss / (i + 1)

                # print log
                post_fix = "\tEpoch:%2d, Iter:%5d, Step:%5d, AvgLoss: %.6f, Loss: %.6f" \
                              % (epoch, i, self.step, avg_loss, loss.item())

                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))

            return avg_loss

    def stream(self, message):
        sys.stdout.write(f"\r{message}")

    def draw(self, data, x, y, ax):
        seaborn.heatmap(data, xticklabels=x, square=True, yticklabels=y,
                        vmin=0.0, vmax=1.0, cbar=False, ax=ax)

    def num_params(self, print_out=True):
        params_requires_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        params_requires_grad = sum([np.prod(p.size()) for p in params_requires_grad])

        parameters = sum([np.prod(p.size()) for p in self.model.parameters()])
        if print_out:
            print('\tTrainable total Parameters: %d' % parameters)
            print('\tTrainable requires_grad Parameters: %d' % params_requires_grad)

    def save_bert_model(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "_ep%d.model" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("\n[+] EP_%d BERT model (%s)" % (epoch, output_path))
        return output_path

    def save_mlm_model(self, epoch, file_path="output/mlm_trained.model"):
        """
        Saving the current MLM model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "_ep%d.model" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("\n[+] EP_%d MLM model (%s)" % (epoch, output_path))
        return output_path

class Paths():
    def __init__(self, output_path):
        self.output_path = output_path
        self.bert_path = f'{output_path}/model_bert'
        self.mlm_path = f'{output_path}/model_mlm'
        self.bert_checkpoints_path = f'{output_path}/bert_checkpoints_path'
        self.runs_path = f'{output_path}/runs'
        self.create_paths()

    def create_paths(self):
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.bert_path, exist_ok=True)
        os.makedirs(self.mlm_path, exist_ok=True)
        os.makedirs(self.bert_checkpoints_path, exist_ok=True)
        os.makedirs(self.runs_path, exist_ok=True)

def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')

def collate_mlm(batch):
    input_lens = [len(x[0]) for x in batch]
    max_x_len = max(input_lens)

    # chars
    instrs_pad = [pad1d(x[0], max_x_len) for x in batch]
    instrs = np.stack(instrs_pad)

    # labels
    labels_pad = [pad1d(x[1], max_x_len) for x in batch]
    labels = np.stack(labels_pad)

    # position
    position = [pad1d(range(1, len + 1), max_x_len) for len in input_lens]
    position = np.stack(position)

    instrs = torch.tensor(instrs).long()
    labels = torch.tensor(labels).long()
    position = torch.tensor(position).long()

    output = {"mlm_input": instrs,
              "mlm_label": labels,
              "input_position": position}

    return output

class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, encoding="utf-8"):
        self.vocab = vocab
        self.num_data = 0
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.corpus = []

        with open(corpus_path, "r", encoding=encoding) as f:
            for line in tqdm.tqdm(f, desc="[+] Loading Dataset", total=self.num_data):
                corpus = (line.split('\t')[2]).replace(" ","_")
                len_tokens = len(corpus.split(','))
                if 5 < len_tokens < hp.enc_maxlen - 5:
                    self.corpus.append(corpus.replace(',', ' '))

            self.num_data = len(self.corpus)
            print ("[+] Number of actual dataset loaded: %d" % self.num_data)

    def __len__(self):
        return self.num_data

    def __getitem__(self, item):
        t = self.corpus[item]
        t_random, t_label = BERTDataset.masking_word(t, wv=self.vocab)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        mlm_input = [self.vocab.sos_index] + t_random + [self.vocab.eos_index]
        mlm_label = [self.vocab.pad_index] + t_label + [self.vocab.pad_index]

        return mlm_input, mlm_label

    @staticmethod
    def masking_word(sentence, wv=None):
        tokens = sentence.split()
        instructions = [instn for instn in tokens]
        output_label = []

        voca_ins = []
        for i, insn in enumerate(instructions):
            if i >= hp.enc_maxlen-3: break
            prob = random.random()
            # randomly choose words to be replaced with a <mask>
            #    15% of the chance (original BERT paper)
            if prob < 0.15:
                prob /= 0.15

                # [80%] token -> mask token
                if prob < 0.8:
                    voca_ins.append(wv.mask_index)
                    #instructions[i] = wv.mask_index

                # [10%] token -> random token
                elif prob < 0.9:
                    voca_ins.append(random.randrange(wv.vocab_size))
                    #instructions[i] = random.randrange(wv.vocab_size)

                # [10%] token -> current token
                else:
                    voca_ins.append(wv.voca_idx(insn))
                    #instructions[i] = wv.voca_idx(insn)

                output_label.append(wv.voca_idx(insn))

            else:
                voca_ins.append(wv.voca_idx(insn))
                output_label.append(0)

        return voca_ins, output_label

def train_model(vocab_path, args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = Paths(args.output_path)

    wv = WordVocab.load_vocab(vocab_path)
    print("[+] Loaded %d vocas from %s" % (wv.vocab_size, vocab_path))
    args.vocab_size = wv.vocab_size

    if args.train_dataset and args.valid_dataset:
        print("[+] Loading Train Dataset", args.train_dataset)
        train_dataset = BERTDataset(args.train_dataset, wv)

        print("[+] Loading Valid Dataset", args.valid_dataset)
        valid_dataset = BERTDataset(args.valid_dataset, wv)

    else:
        dataset = BERTDataset(args.corpus_dataset, wv)
        train_len = int(len(dataset) * hp.train_dataset_ratio)
        valid_len = len(dataset) - train_len
        train_dataset, valid_dataset = \
            random_split(dataset, [train_len, valid_len],
                         generator=torch.Generator().manual_seed(args.seed))
        print ("[+] Split the given dataset (%.2f): training (%d), validation (%d)" \
               % (hp.train_dataset_ratio, train_len, valid_len))

    print("[+] Creating train/validation dataloaders")
    train_data_loader = DataLoader(train_dataset, batch_size=hp.batch_size,
                                   collate_fn=lambda batch: collate_mlm(batch), shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=hp.batch_size,
                                   collate_fn=lambda batch: collate_mlm(batch), shuffle=True)

    print("[+] Building a BERT model")
    bert = BERTEncoder(args=args)

    print("[+] Creating a BERT trainer")
    trainer = BERTTrainer(bert, wv.vocab_size, args.cuda,
                          train_dataloader=train_data_loader,
                          valid_dataloader=valid_data_loader,
                          path=paths)

    print("[+] Start training...")
    trainer.train()



if __name__ == '__main__':
    # Make CUDA report the error when encountered it
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument("-td", "--train_dataset", required=False, type=str,
                        help="train dataset for BERT training")
    parser.add_argument("-vd", "--valid_dataset", required=False, type=str,
                        help="valid dataset to evaluate a train set")
    parser.add_argument("-cd", "--corpus_dataset", required=False, type=str,
                        help="whole corpus for BERT")

    # Path settings
    parser.add_argument("-op", "--output_path", required=False, type=str,
                        help="path for the bert model")
    parser.add_argument("-vp", "--vocab_path", required=False, type=str,
                        help="path of vocab")

    # Others
    parser.add_argument('--seed', type=int, default=99,
                        help="random seed for initialization")
    parser.add_argument('--cuda', type=str, default='0',
                        help="cuda device")

    args = parser.parse_args()

    # Prepare a set of vocabulary if needed
    vocab_path = args.vocab_path

    # python3 bert_mlm.py -cd ./corpus/findutils.corpus.txt
    #        -vp ./vocas/testsuite-all.corpus.vocab
    #        -op ./corpus/output-all
    train_model(vocab_path, args)

