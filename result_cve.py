################################################################
# Practical Binary Code Similarity Detection                   #
#   with BERT-based Transferable Similarity Learning           #
#   (In the 38th Annual Computer Security                      #
#    Applications Conference (ACSAC)                           #
#                                                              #
#  Author: Seonggwan Ahn <sgahn@sor.snu.ac.kr>                 #
#          Dept. of Electrical and Computer Engineering        #
#            @ Seoul National University                       #
#          Hyungjoon Koo <kevin.koo@skku.edu>                  #
#          Dept. of Computer Science and Engineering           #
#            @ Sungkyunkwan University                         #
#                                                              #
#  This file can be distributed under the MIT License.         #
#  See the LICENSE file for details.                           #
################################################################


import argparse
import os
import re
import json
from sklearn.metrics import roc_curve, precision_recall_fscore_support, auc, accuracy_score
from sklearn import metrics
from itertools import combinations_with_replacement, product
from operator import itemgetter
from tqdm import tqdm
from collections import defaultdict

# This is for calculating CVE results


class CveInfo:
    def __init__(self):
        self.vulnerable_funcs = ['tls1_process_heartbeat', 'dtls1_process_heartbeat',\
                            'dtls1_get_message_fragment', 'OBJ_obj2txt', 'ssl3_get_new_session_ticket',\
                            'crypto_recv', 'ctl_putdata', 'configure', 'decode_cell_data']

        self.program_list = {'libav_debug_avconv': ['decode_cell_data'],\
                        'ntp_debug_ntpd': ['crypto_recv', 'ctl_putdata', 'configure'],\
                        'openssl_debug_openssl': ['tls1_process_heartbeat', 'dtls1_process_heartbeat', 'dtls1_get_message_fragment', 'OBJ_obj2txt', 'ssl3_get_new_session_ticket']}


    def make_cve_results(self):
        cve_results = defaultdict(dict)
        for p in self.program_list:
            cve_results[p] = defaultdict(dict)
            cve_results[p]['vul'] = defaultdict(dict)
            cve_results[p]['nonvul'] = defaultdict(dict)
            for f in self.program_list[p]:
                cve_results[p]['vul'][f] = defaultdict(list)
        return cve_results


class CveMetric(CveInfo):
    def __init__(self, source_file, threshold, is_ctr, include_fp):
        super().__init__()
        self.cve_results = self.make_cve_results()
        self.source_file = source_file
        self.threshold = threshold
        self.is_ctr = is_ctr
        self.include_fp = include_fp

    def parse(self):
        """Parsing BinShot result file and making dictionary for CVE detection results per program.
        Return:
            cve_results: (dict) dictionary that contains cve results per program
        """
        with open(self.source_file, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines, desc='parsing file...'):
            true, _, _, sim_score = line.split('\t')[-1].split('///')
            file_info = line.split('\t')[-2].strip()
            if int(float(true)) == 1:
                bn_fn, corpus_comp_opt, query_comp_opt = re.split('(?<=\w):(?=\w)', file_info)
                query_program, query_func = bn_fn.split('@')

            elif int(float(true)) == 0:
                corpus_bn_fn, corpus_comp_opt, query_bn_fn, query_comp_opt = re.split('(?<=\w):(?=\w)', file_info)
                query_program, query_func = query_bn_fn.split('@')

              
            sim_score = float(sim_score)
            pred = 1 if sim_score >= self.threshold else 0
            pred = 1 - pred if self.is_ctr else pred
            ## append 
            # 0: true negative (label pos, pred neg)
            # 1: true positive (label pos, pred pos)
            # 2: false positive (label neg, pred pos)
            # 3: false negative (label neg, pred neg)  
            if int(float(true)) == 1:
                if query_func in self.vulnerable_funcs:
                    if pred == 1:
                        # true positive
                        self.cve_results[query_program]['vul'][query_func][query_comp_opt].append(1)
                    else:
                        # true negative
                        self.cve_results[query_program]['vul'][query_func][query_comp_opt].append(0)

            elif int(float(true)) in [0, -1]:
                if query_func in self.vulnerable_funcs:
                    if pred == 1:
                        # false positive
                        self.cve_results[query_program]['vul'][query_func][query_comp_opt].append(2)
                    else:
                        # false negative
                        self.cve_results[query_program]['vul'][query_func][query_comp_opt].append(3)

                else:
                    if query_comp_opt not in self.cve_results[query_program]['nonvul'][query_func]:
                        self.cve_results[query_program]['nonvul'][query_func] = defaultdict(list)

                    if pred == 1:
                        # false positive
                        self.cve_results[query_program]['nonvul'][query_func][query_comp_opt].append(2)
                    else:
                        # false negative
                        self.cve_results[query_program]['nonvul'][query_func][query_comp_opt].append(3)

        return self.cve_results


    def cleaning_dict(self, vul_dict, non_vul_dict):
        """Convert the results of each function pair into predictions and labels in the units of function(per compiler-optimization)
        Parameters:
            vul_dict : (dict) contains vulnerability functions
            non_vul_dict: (dict) contains non-vulnerability functions
        Return:
            preds, labels : cve detection results in function(per compiler-optimization) unit. each consists of 0, 1
        """
        preds, labels = list(), list()
        new_dict = dict(vul_dict) 
        new_dict.update(non_vul_dict)

        for func in new_dict:
            for comp_opt in new_dict[func]:
                if 1 in new_dict[func][comp_opt]:
                    # if it detected true pair(true positive)
                    if 2 in new_dict[func][comp_opt]:
                        # if it also detected false pair(false positive)
                        if self.include_fp:
                            # don't care false positive
                            pred, label = 1, 1
                        else:
                            # exclude because of false positive
                            pred, label = 0, 1
                    else:
                        # if it detected only true pair(only true positive)
                        pred, label = 1, 1
                else:
                    # if it didn't detect true pair
                    if func not in self.vulnerable_funcs:
                        # since func not in vulnerable_funcs, true pair doesn't exist.
                        if 2 in new_dict[func][comp_opt]:
                            pred, label = 1, 0
                        else:
                            pred, label = 0, 0
                    elif func in self.vulnerable_funcs:
                        # it didn't detect any true pair
                        pred, label = 0, 1

                preds.append(pred)
                labels.append(label)
        
        return preds, labels
   

    def detect(self, vul_dict):
        """Cve detection result per function
        """
        # add varaible for detect_results
        detect_results = defaultdict(dict)
        for func in vul_dict:
            for comp_opt in vul_dict[func]:
                if 1 in vul_dict[func][comp_opt]:
                    if 2 in vul_dict[func][comp_opt]:
                        # mix true positive & false positive
                        if self.include_fp:
                            detect_results[func][comp_opt] = True
                        else:
                            detect_results[func][comp_opt] = False
                    else:
                        # only true positive
                        detect_results[func][comp_opt] = True
                else:
                    # no true positive
                    detect_results[func][comp_opt] = False
        return detect_results


def compute_metric(pred, label, avg='binary'):
    """Compute metric using sklearn
    """
    precision, recall, f1, _ = precision_recall_fscore_support(label, pred, average=avg)
    acc = accuracy_score(label, pred)
    fpr, tpr, _ = roc_curve(label, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': auc
    }


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_file", required=True, type=str, help="result file")
    parser.add_argument("-v", "--save_dir", required=True, type=str, help="save directory")
    parser.add_argument("-t", "--threshold", required=True, type=float, help="threshold")
    parser.add_argument("-c", "--is-contrastive", action='store_true', help='if it is binshot-ctr')
    parser.add_argument("-f", "--include-fp", action='store_true', help='if it include false positive as positive detection')
    
    args = parser.parse_args()
    source_file = args.source_file
    save_dir = args.save_dir
    thr = args.threshold
    is_ctr = args.is_contrastive
    include_fp = args.include_fp


    cve_metric = CveMetric(source_file, thr, is_ctr, include_fp)
    cve_results = cve_metric.parse()

    fw = open(os.path.join(save_dir, 'cve_result.txt'), 'w')
    for p in cve_results:
        detect_result = cve_metric.detect(cve_results[p]['vul'])
        preds, labels = cve_metric.cleaning_dict(cve_results[p]['vul'], cve_results[p]['nonvul'])
        p_result = compute_metric(preds, labels)
        fw.write(f'[{p}] : {p_result}\n')
        fw.write(f'[vulnerable function detection results]\n')
        for fname, dresult in detect_result.items():
            fw.write(f'{fname}: {dresult}\n')
        fw.write('\n\n')
    fw.close()
