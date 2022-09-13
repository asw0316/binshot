################################################################
# Practical Binary Code Similarity Detection                   #
#   with BERT-based Transferable Similarity Learning           #
#   (In the 38th Annual Computer Security                      #
#    Applications Conference (ACSAC)                           #
#                                                              #
#  Author: Sunwoo Ahn <swahn@sor.snu.ac.kr>                    #
#          Dept. of Electrical and Computer Engineering        #
#            @ Seoul National University                       #
#          Hyungjoon Koo <kevin.koo@skku.edu>                  #
#          Dept. of Computer Science and Engineering           #
#            @ Sungkyunkwan University                         #
#                                                              #
#  This file can be distributed under the MIT License.         #
#  See the LICENSE file for details.                           #
################################################################

import os, sys
import statistics
import argparse
import random
import tqdm
import traceback
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

from bert_mlm import Linear, MultiheadAttention, Attention
from bert_mlm import Convolution, PositionWiseFeedForward
from bert_mlm import EncoderPrenet, BERTEncoder, BertAdam, optim4GPU

import hparams as hp
from util import compute_prediction_metric
from util import write_metrics, write_pred_results
from voca import WordVocab

class SimilarityModel(nn.Module):
    """
    Similarity Model
    """

    def __init__(self, bert: BERTEncoder):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert

        self.l2_dist = nn.MSELoss(reduction="none")
        self.linear = nn.Linear(hp.num_hidden, 1)

    def forward(self, x1, x1_pos, x2, x2_pos, mode=0):
        if (mode==0) or (mode==1):
            x1, x1_attn_list, x1_xs = self.bert(x1, x1_pos)
            x2, x2_attn_list, x2_xs = self.bert(x2, x2_pos)

            x1 = x1[:,0,:]
            x2 = x2[:,0,:]

        if mode==1:
            return x1, x2

        x = self.l2_dist(x1, x2)
        output = self.linear(x)

        if (mode==0) or (mode==2):
            return torch.squeeze(output, 1)

    def init_model(self):
        un_init = ['bert.embed.weight', 'bert.pos_emb.weight']
        for n, p in self.named_parameters():
            if n not in un_init and p.dim() > 1:
                nn.init.xavier_uniform_(p)



class SimilarityTrainer():

    def __init__(self, bert: BERTEncoder, vocab_size: int, cuda,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader = None,
                 test_dataloader: DataLoader = None, ft_model_path: str = '',
                 log_freq: int = hp.log_freq, global_step=0, path=None):
        """
        :param bert: Similarity model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param valid_dataloader: valid dataset data loader
        :param test_dataloader: test dataset data loader
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param log_freq: logging frequency of the batch iteration
        """

        self.step = global_step
        self.path = path

        # Setup a cuda device for SimModel training
        has_cuda = torch.cuda.is_available()
        device_loc = "cuda:"+cuda if has_cuda else "cpu"
        self.device = torch.device(device_loc)

        # This BERT model will be saved every epoch
        self.bert = bert

        # Initialize a Similarity Model with a BERT model
        self.model = None
        if ft_model_path != "":
            self.model = torch.load(ft_model_path, map_location=device_loc).to(self.device) \
		if torch.cuda.device_count() == 1 else torch.load(ft_model_path).to(self.device)
        else:
            self.model = SimilarityModel(bert).to(self.device)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        if self.train_data is None:
            total_steps = 0
        else:
            total_steps = hp.epochs * len(self.train_data)
        self.optimer = optim4GPU(self.model, total_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.BCEWithLogitsLoss()

        # Writer
        self.log_freq = log_freq

        self.num_params()

    def set_test_data(self, tt):
        self.test_data = tt

    def initialize_log(self, fp):
        if os.path.isfile(fp):
            os.remove(fp)

    def train(self):
        train_metric_fp = f'{self.path.train_metric_fp}'
        self.initialize_log(train_metric_fp)

        try:
            for epoch in range(hp.epochs):
                # Setting a tqdm progress bar
                data_iter = tqdm.tqdm(enumerate(self.train_data),
                                      desc="[+] EP_%s (%d)" % ("train", epoch),
                                      total=len(self.train_data),
                                      bar_format="{l_bar}{r_bar}")

                running_loss = 0
                preds, labels = np.array([]), np.array([])
                for i, data in data_iter:
                    self.step += 1

                    # 0. batch_data will be sent into the device (GPU or cpu)
                    data = {key: value.to(self.device) for key, value in data.items() if key != 'line'}

                    # 1. forward masked_lm model
                    sim_output = \
                        self.model.forward(data["f1_input"],
                        data["f1_position"], data["f2_input"],
                        data["f2_position"])

                    # 2. NLLLoss of predicting masked token word
                    self.optimer.zero_grad()
                    loss = self.criterion(sim_output, data["label"])

                    # 3. backward and optimization only in train
                    loss.backward()
                    self.optimer.step()

                    # loss
                    running_loss += loss.item()
                    avg_loss = running_loss / (i + 1)

                    # Evaluation of test results
                    preds_batch = torch.round(torch.sigmoid(sim_output)).cpu().detach().numpy()
                    labels_batch = data["label"].cpu().detach().numpy()

                    preds = np.concatenate((preds, preds_batch), axis=-1)
                    labels = np.concatenate((labels, labels_batch), axis=-1)

                    # print log
                    if i % self.log_freq == 0:
                        result_acc = compute_prediction_metric(preds, labels, avg='binary')
                        post_fix = "\tIter:%5d, Step:%5d, AvgLoss: %.6f, Loss: %.6f, Acc: %.3f, " \
                                   "Precision: %.3f, Recall: %.3f, F1: %.3f, AUC: %.3f" \
                                   % (i, self.step, avg_loss, loss.item(), result_acc['accuracy'],
                                      result_acc['precision'], result_acc['recall'],
                                      result_acc['f1'], result_acc['auc'])

                        data_iter.write(str(post_fix))

                valid_loss = self.validation(epoch)
                self.save_bert_model(epoch, f"{self.path.sim_path}/bert")
                self.save_sim_model(epoch, f"{self.path.sim_path}/sim")
                print(f"EP_{epoch}, train_avg_loss={avg_loss}, valid_avg_loss={valid_loss}")

        except BaseException:
            traceback.print_exc()

    def validation(self, epoch):
        self.model.eval()

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.valid_data),
                              desc="[+] EP_%s (%d)" % ("valid", epoch),
                              total=len(self.valid_data),
                              bar_format="{l_bar}{r_bar}")

        running_loss = 0
        preds, labels = np.array([]), np.array([])
        with torch.no_grad():
            for i, data in data_iter:
                self.step += 1

                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items() if key != 'line'}

                # 1. forward masked_lm model
                sim_output = \
                    self.model.forward(data["f1_input"],
                    data["f1_position"], data["f2_input"],
                    data["f2_position"])

                # 2. NLLLoss of predicting masked token word
                loss = self.criterion(sim_output, data["label"])

                # loss
                running_loss += loss.cpu().detach().numpy()
                avg_loss = running_loss / (i + 1)

                # Evaluation of test results
                preds_batch = torch.round(torch.sigmoid(sim_output)).cpu().detach().numpy()
                labels_batch = data["label"].cpu().detach().numpy()

                preds = np.concatenate((preds, preds_batch), axis=-1)
                labels = np.concatenate((labels, labels_batch), axis=-1)

                # write log
                if i % self.log_freq == 0:
                    acc_result = compute_prediction_metric(preds, labels, avg='binary')
                    post_fix = "\tIter:%5d, Step:%5d, AvgLoss: %.6f, Loss: %.6f, Acc: %.3f, " \
                               "Precision: %.3f, Recall: %.3f, F1: %.3f, AUC: %.3f" \
                               % (i, self.step, avg_loss, loss.item(), acc_result['accuracy'],
                                  acc_result['precision'], acc_result['recall'],
                                  acc_result['f1'], acc_result['auc'])

                    data_iter.write(str(post_fix))

            return avg_loss

    def test(self):
        self.model.eval()

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.test_data),
                              desc="[+] EP_test ",
                              total=len(self.test_data),
                              bar_format="{l_bar}{r_bar}")

        test_metric_fp = f'{self.path.test_metric_fp}'
        test_pred_fp = f'{self.path.test_pred_fp}'
        self.initialize_log(test_metric_fp)
        self.initialize_log(test_pred_fp)

        running_loss = 0
        preds, labels , scores= np.array([]), np.array([]), np.array([])
        lines = []
        with torch.no_grad():
            for i, data in data_iter:
                self.step += 1
                lines += data["line"]

                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items() if key != 'line'}

                # 1. forward masked_lm model
                sim_output = \
                    self.model.forward(data["f1_input"],
                    data["f1_position"], data["f2_input"],
                    data["f2_position"])
                preds_batch = torch.round(torch.sigmoid(sim_output)).cpu().detach().numpy()

                # 2. NLLLoss of predicting masked token word
                loss = self.criterion(sim_output, data["label"])
                scores = np.concatenate((scores,torch.sigmoid(sim_output).cpu().detach().numpy().reshape(-1)),\
                                        axis=-1)

                # loss
                running_loss += loss.cpu().detach().numpy()
                avg_loss = running_loss / (i + 1)

                # Evaluation of test results
                labels_batch = data["label"].cpu().detach().numpy()

                preds = np.concatenate((preds, preds_batch), axis=-1)
                labels = np.concatenate((labels, labels_batch), axis=-1)

            # print log
            result_acc = compute_prediction_metric(preds, labels, avg='binary')
            post_fix = "\tAcc: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f, AUC: %.3f" \
                       % (result_acc['accuracy'], result_acc['precision'],
                          result_acc['recall'], result_acc['f1'], result_acc['auc'])
            print(post_fix)
            write_pred_results(test_pred_fp, preds, labels, lines, scores)


    def stream(self, message):
        sys.stdout.write(f"\r{message}")

    def num_params(self, print_out=True):
        params_requires_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        params_requires_grad = sum([np.prod(p.size()) for p in params_requires_grad])

        parameters = sum([np.prod(p.size()) for p in self.model.parameters()])
        if print_out:
            print('Trainable total Parameters: %d' % parameters)
            print('Trainable requires_grad Parameters: %d' % params_requires_grad)

    def save_bert_model(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current finetuned BERT model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "_ep%d.model" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def save_sim_model(self, epoch, file_path="output/sim_trained.model"):
        """
        Saving the current Similarity model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "_ep%d.model" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


class Paths():
    def __init__(self, output_path, result_path):
        self.output_path = output_path
        self.bert_path = f'{output_path}/model_bert'
        self.sim_path = f'{output_path}/model_sim'
        self.bert_checkpoints_path = f'{output_path}/bert_checkpoints_path'
        self.runs_path = f'{output_path}/runs'
        self.train_metric_fp = f'{output_path}/metric.train.{result_path}'
        self.test_metric_fp = f'{output_path}/metric.test.{result_path}'
        self.test_pred_fp = f'{output_path}/pred.test.{result_path}'
        self.create_paths()

    def create_paths(self):
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.bert_path, exist_ok=True)
        os.makedirs(self.sim_path, exist_ok=True)
        os.makedirs(self.bert_checkpoints_path, exist_ok=True)
        os.makedirs(self.runs_path, exist_ok=True)

    def set_result_path(self, result_path):
        output_path = self.output_path
        self.train_metric_fp = f'{output_path}/metric.train.{result_path}'
        self.test_metric_fp = f'{output_path}/metric.test.{result_path}'
        self.test_pred_fp = f'{output_path}/pred.test.{result_path}'


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')

def collate_sim(batch):
    f1_input_lens = [len(x[0]) for x in batch]
    f2_input_lens = [len(x[1]) for x in batch]
    max_x_len = max(f1_input_lens+f2_input_lens)

    # padded
    f1_instrs = [pad1d(x[0], max_x_len) for x in batch]
    f2_instrs = [pad1d(x[1], max_x_len) for x in batch]
    f1_instrs = np.stack(f1_instrs)
    f2_instrs = np.stack(f2_instrs)

    # labels
    labels_pad = [x[2] for x in batch]
    labels = np.stack(labels_pad)

    # position
    f1_position = []
    f2_position = []
    for f1_len, f2_len in zip(f1_input_lens, f2_input_lens):
        f1_position.append(pad1d(range(1, f1_len + 1), max_x_len))
        f2_position.append(pad1d(range(1, f2_len + 1), max_x_len))
    f1_position = np.stack(f1_position)
    f2_position = np.stack(f2_position)

    f1_instrs = torch.tensor(f1_instrs).long()
    f2_instrs = torch.tensor(f2_instrs).long()

    labels = torch.tensor(labels).float()

    f1_position = torch.tensor(f1_position).long()
    f2_position = torch.tensor(f2_position).long()

    # raw lines for logging
    line = [x[3] for x in batch]

    output = {"f1_input": f1_instrs,
              "f2_input": f2_instrs,
              "label": labels,
              "f1_position": f1_position,
              "f2_position": f2_position,
              "line":line}

    return output


class SimDataset(Dataset):
    def __init__(self, corpus_path, vocab, encoding="utf-8"):
        self.vocab = vocab
        self.num_data = 0
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.corpus = []

        with open(corpus_path, "r", encoding=encoding) as f:
            for line in tqdm.tqdm(f, desc="Loading Dataset", total=self.num_data):
                f1_corpus, f2_corpus, gt, label, = line.replace('\n', '').split('\t')
                f1_corpus = f1_corpus.replace(" ","_")
                f2_corpus = f2_corpus.replace(" ","_")

                #preprocessing
                label = int(label)
                len_tokens_f1 = len(f1_corpus.split(','))
                len_tokens_f2 = len(f2_corpus.split(','))

                if "cve" in corpus_path:
                    if (5 < len_tokens_f1)\
                            and (5 < len_tokens_f2):
                        self.corpus.append((f1_corpus, f2_corpus, label, line))
                else:
                    if (5 < len_tokens_f1 < hp.enc_maxlen -5)\
                            and (5 < len_tokens_f2 < hp.enc_maxlen - 5):
                        self.corpus.append((f1_corpus, f2_corpus, label, line))

            self.num_data = len(self.corpus)

    def __len__(self):
        return self.num_data

    def __getitem__(self, item):
        f1_instr, f2_instr, label, line, = self.corpus[item]
        f1_instr_idx = SimDataset.random_word(f1_instr, wv=self.vocab)
        f2_instr_idx = SimDataset.random_word(f2_instr, wv=self.vocab)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        f1 = [self.vocab.sos_index] + f1_instr_idx + [self.vocab.eos_index]
        f2 = [self.vocab.sos_index] + f2_instr_idx + [self.vocab.eos_index]

        return f1, f2, label, line.replace('\n', '')

    @staticmethod
    def random_word(sentence, wv=None):
        tokens = sentence.split(',')
        instructions = [instn for instn in tokens]

        voca_ins = []
        for i, insn in enumerate(instructions):
            if i >= hp.enc_maxlen-3: break
            voca_ins.append(wv.voca_idx(insn))

        return voca_ins

def run_model(bert_model_path, vocab_path, corpus_paths, output_path,
              result_path, cuda, seed=99):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    has_cuda = torch.cuda.is_available()
    device_loc = "cuda:"+cuda if has_cuda else "cpu"

    paths = Paths(output_path, result_path)

    wv = WordVocab.load_vocab(vocab_path)
    print("[+] Loaded %d vocas from %s" % (wv.vocab_size, vocab_path))

    if args.ft_model_path == '':
        train_dataset = SimDataset(corpus_paths['train'], wv)
        valid_dataset = SimDataset(corpus_paths['valid'], wv)

        print("[+] Creating Dataloaders")
        train_data_loader = DataLoader(train_dataset, batch_size=hp.batch_size,
                                       collate_fn=lambda batch: collate_sim(batch), shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=hp.batch_size,
                                       collate_fn=lambda batch: collate_sim(batch), shuffle=True)

        print("[+] Loading a Pre-traind model")
        pretrained = torch.load(bert_model_path, map_location=device_loc) \
		if torch.cuda.device_count() == 1 else torch.load(bert_model_path)
        pretrained.eval()

        print("[+] Creating a Similarity Trainer")
        trainer = SimilarityTrainer(pretrained, wv.vocab_size, cuda,
                                    train_dataloader=train_data_loader,
                                    valid_dataloader=valid_data_loader,
                                    ft_model_path=args.ft_model_path,
                                    path=paths)

        print("[+] Start training...")
        trainer.train()

    else:
        print("[+] Loading a Pre-traind model")
        pretrained = torch.load(bert_model_path, map_location=device_loc) \
		if torch.cuda.device_count() == 1 else torch.load(bert_model_path)
        pretrained.eval()

    test_dataset = SimDataset(corpus_paths['test'], wv)

    print("[+] Creating Dataloaders")
    test_data_loader = DataLoader(test_dataset, batch_size=hp.batch_size,
                                  collate_fn=lambda batch: collate_sim(batch), shuffle=False)

    if args.ft_model_path != '':
        print("[+] Creating a Similarity Trainer")
        trainer = SimilarityTrainer(pretrained, wv.vocab_size, cuda,
                                    train_dataloader=None,
                                    test_dataloader=test_data_loader,
                                    ft_model_path=args.ft_model_path,
                                    path=paths)
    else:
        trainer.set_test_data(test_data_loader)

    print("[+] Start testing...")
    trainer.test()

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    parser = argparse.ArgumentParser()

    # Dataset paths
    parser.add_argument("-tn", "--train_dataset", required=False, type=str,
                        help="train dataset")
    parser.add_argument("-vd", "--valid_dataset", required=False, type=str,
                        help="valid dataset")
    parser.add_argument("-tt", "--test_dataset", required=False, type=str,
                        help="test dataset")

    # Input paths
    parser.add_argument("-vp", "--vocab_path", required=False, type=str,
                        help="vocabulary (see voca.py)")
    parser.add_argument("-bm", "--bert_model_path", required=False, type=str,
                        help="pretraining model")
    parser.add_argument("-fm", "--ft_model_path", required=False, type=str,
                        default='', help="fine-tuning model")

    # Output paths
    parser.add_argument("-op", "--output_path", required=False, type=str,
                        help="output path")
    parser.add_argument("-r", "--result_path", required=False, type=str,
                        default='', help="result file path for evaluation")

    # Others
    parser.add_argument('--seed', type=int, default=99,
                        help="random seed for initialization")
    parser.add_argument('--cuda', type=str, default='0',
                        help="cuda device")

    args = parser.parse_args()

    corpus_paths = {
        'train': args.train_dataset,
        'valid': args.valid_dataset,
        'test': args.test_dataset,
    }
    
    device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    # Example:
    # python3 deepsemantic-binsim.py -bm models/pretrain/model_bert/bert_ep19.model
    #       [-fm modles/downstream/model_sim/sim_ep19.model] (TEST ONLY!)
    #       -v corpus/pretrain.findutils.corpus.voca
    #       -o models/downstream
    #       -r findutils
    #       -tn corpus/binsim.findutils.train.corpus.txt
    #       -vd corpus/binsim.findutils.valid.corpus.txt
    #       -tt corpus/binsim.findutils.test.corpus.txt
    run_model(args.bert_model_path, args.vocab_path, corpus_paths, args.output_path,
              args.result_path, args.cuda, args.seed)
