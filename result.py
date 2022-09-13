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


def parse(source_file):
    comp_list = ['clang', 'gcc']
    opt_list = ['O'+str(i) for i in range(4)]

    table_list = [c+o for c, o in list(product(comp_list, opt_list, repeat=1))]
    pair_list = list(combinations_with_replacement(table_list, 2))
    pair_list.sort(key=itemgetter(0))
    result = {p1+'_'+p2: dict() for (p1, p2) in pair_list}
    result['total'] = dict()
    
    for key in result:
        result[key]['labels'] = list()
        result[key]['preds'] = list()

    with open(source_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        file_info, res = line.split('\t')[-2].strip(), line.split('\t')[-1].strip()
        true, pred = res.split('///')[0], res.split('///')[1]
        
        if int(true) == 0:
            _, comp_opt_1, _, comp_opt_2 = re.split('(?<=\w):(?=\w)', file_info)
            comp_opts = [comp_opt_1, comp_opt_2]

        elif int(true) == 1:
            bn_fn, *comp_opts = re.split('(?<=\w):(?=\w)', file_info)
            assert len(comp_opts) == 2
        
        result['total']['labels'].append(int(float(true)))
        result['total']['preds'].append(int(float(pred)))
        
        result['_'.join(sorted(comp_opts))]['labels'].append(int(float(true)))
        result['_'.join(sorted(comp_opts))]['preds'].append(int(float(pred)))
    
    return result


def compute_metric(pred, label, avg='binary'):
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
    parser.add_argument("-s", "--source_file", required=True, type=str, help="result file or directory")
    parser.add_argument("-v", "--save_dir", required=True, type=str, help="save directory")
    args = parser.parse_args()
    source_file = args.source_file
    save_dir = args.save_dir

    results = parse(source_file)

    for key in results:
        results[key] = compute_metric(results[key]['preds'], results[key]['labels'])
   
    # save file in json format
    prefix = source_file.split('/')[-1].split('.')[2]
    with open(os.path.join(save_dir, f'binshot_{prefix}_result_comp_opt_pair.json'), 'w') as json_wrt:
        json.dump(results, json_wrt, indent=4, sort_keys=True)
        
