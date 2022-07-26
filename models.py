import math
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_

import hparams as hp

#####
# Define the following models:
#   Linear
#   MultiheadAttention
#   Attention
#   Convolution
#   PositionWiseFeedForward
#   EncoderPrenet
#   BERTEncoder
#   BERTAdam
#####

class Linear(nn.Module):
    """
    Linear module class
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
    Multihead attention module class (dot attention)
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
            attn = attn * query_mask

        # Dropout
        self.attention = self.attn_dropout(attn).view(key.size(0)//4, 4, -1, key.size(1))

        # Get a context vector
        result = torch.bmm(attn, value)

        return result, self.attention


class Attention(nn.Module):
    """
    Attention module class
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
    Convolution module class
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
    Position-wise Feed-Forward Neural Network (FFNN) class
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
    Pre-network module for the Encoder class
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
    BERT encoder class
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
        self.layers = self.__clones(Attention(), self.num_attn_layers)
        self.pwffns = self.__clones(PositionWiseFeedForward(), self.num_attn_layers)

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

    def __clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def __get_sinusoid_encoding_table(self, n_position, d_hid, padding_idx=None):
        ''' Sinusoid position encoding table '''

        def cal_angle(position, hid_idx):  # 0-255
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):  # 0-1023
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array(
            [get_posi_angle_vec(pos_i) for pos_i in range(n_position)])  # 1024，256  循环1-1024  1024次

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.

        return torch.FloatTensor(sinusoid_table)


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
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        assert lr > 0.0, "Learning rate: %f - should be > 0.0" % (lr)
        assert schedule in SCHEDULES, "Invalid schedule : %s" % (schedule)
        assert 0.0 <= warmup < 1.0 or warmup == -1.0, \
            "Warmup %f - should be in 0.0 ~ 1.0 or -1 (no warm up)" % (warmup)
        assert 0.0 <= b1 < 1.0, "b1: %f - should be in 0.0 ~ 1.0" % (b1)
        assert 0.0 <= b2 < 1.0, "b2: %f - should be in 0.0 ~ 1.0" % (b2)
        assert e > 0.0, "epsilon: %f - should be > 0.0" % (e)
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
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if not state:
                    state['step'] = 0
                    # Exponential moving average of gradient values 平均梯度的指数移动
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values   平均平方梯度的指数移动
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient  衰减第一和第二时刻的运行平均系数
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
