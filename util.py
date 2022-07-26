import os
import sys
import pickle
import logging
from bz2 import BZ2File
import subprocess
import platform
import numpy as np

from sklearn import metrics

# http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
class ProgressBar():
    DEFAULT_BAR_LENGTH = 50
    DEFAULT_CHAR_ON  = '>'
    DEFAULT_CHAR_OFF = ' '

    def __init__(self, end, start=0, name='N/A'):
        self.end    = end
        self.start  = start
        self.name = name
        self._barLength = self.__class__.DEFAULT_BAR_LENGTH

        self.setLevel(self.start)
        self._plotted = False

    def setLevel(self, level):
        self._level = level
        if level < self.start:  self._level = self.start
        if level > self.end:    self._level = self.end

        self._ratio = float(self._level - self.start) / float(self.end - self.start)
        self._levelChars = int(self._ratio * self._barLength)

    def plotProgress(self):
        tab = '\t'
        sys.stdout.write("\r%s%3i%% [%s%s] (%s)" %(
            tab*1 + '  ', int(self._ratio * 100.0),
            self.__class__.DEFAULT_CHAR_ON  * int(self._levelChars),
            self.__class__.DEFAULT_CHAR_OFF * int(self._barLength - self._levelChars),
            self.name
        ))
        sys.stdout.flush()
        self._plotted = True

    def setAndPlot(self, level):
        oldChars = self._levelChars
        self.setLevel(level)
        if (not self._plotted) or (oldChars != self._levelChars):
            self.plotProgress()

    def __add__(self, other):
        assert type(other) in [float, int], "can only add a number"
        self.setAndPlot(self._level + other)
        return self

    def __sub__(self, other):
        return self.__add__(-other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__add__(-other)

    def finish(self):
        sys.stdout.write("\n")

def demangle(f):
    try:
        import cxxfilt
        demangled = cxxfilt.demangle(f)
    except:
        #logging.warning("Failed to demangle the function name: %s" % f)
        demangled = f

    return demangled

def is_elf(f):
    # Check if the magic number is "\x7F ELF"
    return open(f, 'rb').read(4) == '\x7f\x45\x4c\x46'

def run_cmd(cmd):
    logging.info("Run the command: %s" % (' '.join(cmd)))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, _ = p.communicate()
    p.wait()

def is_output(outs):
    """ Return False if there is any output missing"""
    for out in outs:
        if not os.path.exists(out):
            return False
        else:
            logging.info("[-] Found: %s (To regenerate, remove it!)" % out)

    return True

def read_symbols(function_boundaries_out):
    all_func_symbols = {}

    import debugInfo_pb2
    symbol_info = debugInfo_pb2.SymbolInfo()
    symbol_info.ParseFromString(BZ2File(function_boundaries_out, "rb").read())

    # Multiple function symbols can be defined across different executables
    fn_ctr = 0
    for fn in symbol_info.funinfo:
        if fn.name not in all_func_symbols:
            all_func_symbols[fn.name] = []
        all_func_symbols[fn.name].append((fn.elf, fn.start, fn.end))
        fn_ctr += 1
        # all_func_symbols[fn.name] = (fn.elf, fn.start, fn.end, fn.srcline)

    logging.info("\tLoaded function symbols: %d (%d)" % (len(all_func_symbols), fn_ctr))
    return all_func_symbols

def load_from_dmp(dmp_path):
    """
    Dump all function information collected from IDA Pro
    Each function represents an instance of class unit.IDA_Function()
    :param dmp_path:
    :return:
    """
    functions = dict()
    dmp_file = BZ2File(dmp_path, 'rb')
    cnt = 0
    while True:
        try:

            major_ver, _, _ = platform.python_version_tuple()
            if major_ver == '2':
                import cPickle
                F = cPickle.load(dmp_file)
                #F = pickle.load(dmp_file)
            if major_ver == '3':
                F = pickle.load(dmp_file, encoding='latin1')

            if not F:
                break
            functions[F.start] = F
            cnt += 1

        except MemoryError:
            logging.error('Memory error reading at Function 0x%08X after loading %d functions'
                          % (F.addr, cnt))
            pass

    dmp_file.close()
    return functions

def load_from_json(json_path, is_dump=False):
    """
    The json format has a key of 'filename' and a value of all function info dictionary
    Each function info has a key of 'func_index' (i.e., F12) and a dictionary value of
        i) basic info: fn_[idx, start, end, size, name, num_bbs, num_ins]
        ii) call graph info: num_ref_[tos, froms], ref_tos_by_[call, jump, data], ref_froms
        iii) function signature info: fn_num_imms, is_recursive, glibc_funcs,
                                      str_refs, num_glibc_funcs, fn_imms, fn_num_imms and
        iv) basic block info: bb_info where
            each basic block info (bb_info) has a key of 'bb_index' (i.e., F12_B3) and
            a dictionary value of bb_[idx, start, end, size, num_ins], and ins_info where
                each instruction info (ins_info) has a key of 'ins_index' (i.e., F12_B3_I2)
                a dictionary value of ins_[idx, start, end, size, opcode, operands, normalized, imms],
                has_[imms, ref_string, glibc_call], and ref_string
    """
    import json
    json_txt = ''
    with open(json_path, "r") as f:
        json_data = json.load(f)
        bin_names = json_data.keys()
        for bin_name in sorted(bin_names):
            json_txt += "%s\n" % bin_name
            bin_data = json_data[bin_name]
            functions = sorted(bin_data.keys())
            for func in functions:
                json_txt += "\t%s\n" % (func)
                # json_data[bin_name][func]
                func_data = bin_data[func]
                func_attrs = sorted(func_data.keys())
                for func_attr in func_attrs:
                    if func_attr == 'bbs_info':
                        bbs = sorted(func_data[func_attr].keys())
                        for bb in bbs:
                            json_txt += "\t\t%s\n" % bb
                            # json_data[bin_name][func][func_attr][bb]
                            bb_data = func_data[func_attr][bb]
                            bb_attrs = sorted(bb_data.keys())
                            for bb_attr in bb_attrs:
                                if bb_attr == 'ins_info':
                                    instns = sorted(bb_data[bb_attr].keys())
                                    for instn in instns:
                                        json_txt += "\t\t\t%s\n" % instn
                                        # json_data[bin_name][func][func_attr][bb][bb_attr][instn]
                                        insn_data = bb_data[bb_attr][instn]
                                        instn_attrs = sorted(insn_data.keys())
                                        for instn_attr in instn_attrs:
                                            json_txt += "\t\t\t\t%s: %s\n" % (instn_attr, insn_data[instn_attr])
                                else:
                                    json_txt += "\t\t\t%s: %s\n" % (bb_attr, bb_data[bb_attr])
                    else:
                        json_txt += "\t\t%s:%s\n" % (func_attr, func_data[func_attr])

    if is_dump:
        with open(json_path + '.txt', "w") as g:
            g.write(json_txt)

    return json_data

def compute_prediction_metric(pred, obsv, avg='binary'):
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(obsv, pred,
                                                                       average=avg)
    acc = metrics.accuracy_score(obsv, pred)
    fpr, tpr, thresholds = metrics.roc_curve(obsv, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'fpr':fpr,
        'tpr':tpr,
        'auc':auc
    }

def write_metrics(fp, metric):
    metric_names= ['accuracy', 'f1', 'precision', 'recall', 'fpr', 'tpr', 'auc']
    with open(fp, 'a') as f:
        for mn in metric_names:
            output = metric[mn][1:-1]if mn == 'fpr' or mn == 'tpr' else metric[mn]
            f.write('{},'.format(output))
        f.write("\n")

def write_pred_results(fp, y_pred, y_true, lines, score):
    with open(fp, 'w') as f, open(fp+"_all", 'w') as f2:
        for p, t, l,s in zip(y_pred, y_true, lines,score):
            if not t == p:
                f.write('{}///{}///{}///{}\n'.format(l, p, t,s))
            f2.write('{}///{}///{}///{}\n'.format(l, p, t,s))

