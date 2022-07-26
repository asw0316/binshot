import os, sys, resource, copy
import unit
import util
import pickle
import argparse
import logging
import tqdm

import random
import normalize
import multiprocessing
from multiprocessing import Pool
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from itertools import combinations

def run_normalization(target, normalization_level=3):
    file_name = os.path.basename(target)
    ida_dmp_path = target + ".dmp.gz"
    _, arch, compiler, optlevel = file_name.rsplit("-", 3)
    bin_info = unit.Binary_Info(target)
    bin_info.compiler_info_label = compiler
    bin_info.opt_level_label = optlevel

    # [Step 1] Loading an IDA dump file
    print("\t[+] Loading %s..." % ida_dmp_path)
    nn = normalize.Normalization(util.load_from_dmp(ida_dmp_path))

    # [Step 2] Initialize (must be done before normalization)
    nn.build_bininfo(bin_info)

    # [Step 3] Disassemble and normalize all instructions: pre-processing
    nn.disassemble_and_normalize_instructions(normalization_level=normalization_level)

    # [Step 4] Generate data for machine learning
    corpus_ctr, corpus_data, corpus_voca = nn.generate_learning_data()
    return nn, corpus_ctr, corpus_data, corpus_voca



def corpus_generator(target, pkl_dir, dmp_level=None, write_json=False,
                     normalization_level=3):
    # Assume the given file name the following format
    #   [file_name]-[arch]-[compiler]-[optlevel]
    #   e.g., vsftpd-amd64-clang-O0
    file_name = os.path.basename(target)
    pkl_dmp_path = pkl_dir + file_name + ".pkl"

    corpus_ctr = 0
    corpus_data = ''
    corpus_voca = Counter()

    if not write_json and not dmp_level and os.path.isfile(pkl_dmp_path):
        print("\t[+] Loading %s..." % pkl_dmp_path)
        try:
            BS = pickle.load(open(pkl_dmp_path, 'rb'))
            if BS.compiler and BS.opt_level:
                label = BS.compiler + ' ' + BS.opt_level + '\n'
            else:
                label = "N/A\n"
            for fs in BS.fns_summaries:
                if not fs.is_linker_func:
                    normalized_instrs = [x for x in filter(None, fs.normalized_instrs)]
                    corpus_data += '\t'.join([BS.bin_name, fs.fn_name,
                                              ','.join(normalized_instrs), label])
                    corpus_voca += Counter(normalized_instrs)
                    corpus_ctr += 1
        except:
            print("\t[-] Failed %s..." % pkl_dmp_path)
    else:
        nn, corpus_ctr, corpus_data, corpus_voca = \
            run_normalization(target, normalization_level=normalization_level)

        # By default, we generate a pickle per binary to reduce further computation
        BS = nn.pickle_dump(unit.BinarySummary())
        pickle.dump(BS, open(pkl_dmp_path, 'wb'))

        # Save memory
        del nn

    return file_name, corpus_ctr, corpus_data, corpus_voca

class CorpusGenerator():
    """
    Class for deep learning data generation that holds various statistics
        requires the original binary and its information dumped from IDA
        (TODO: binary information can be obtained from other analysis tools like radare2)
    """
    def __init__(self, target):
        self.target = target

        # Corpus information
        self.num_corpus = 0
        self.corpus = []
        self.jsons = {}
        self.vocas = Counter()

        self.targets = []
        self.__collect_targets()
        
    def __collect_targets(self):
        """
        Assume all files ending with [0-3] are a correct ELF form
        """
        if os.path.isfile(self.target):
            target_file = self.target
            self.targets.append(target_file)
        elif os.path.isdir(self.target):
            target_dir = self.target
            for ea in os.listdir(target_dir):
                if ea.endswith('.dmp.gz') or ea.endswith('.id0') or ea.endswith('id1')\
                  or ea.endswith('id2') or ea.endswith('.nam') or ea.endswith('.til'):
                    continue
                # The following condition might be changed upon dataset names
                elif ea[-1] in ['0', '1', '2', '3']:
                    target_path = os.path.join(target_dir, ea)
                    ida_dmp_path = target_path + ".dmp.gz"
                    if not os.path.exists(ida_dmp_path):
                        print("[-] Terminating: %s has not been found!" % \
                              (ida_dmp_path))
                        sys.exit(-1)
                    else:
                        self.targets.append(target_path)
        else:
            print ("No such a file or directory...!")
            sys.exit(-1)

        print ("Number of targets to process: %d" % len(self.targets))

    def run_slow(self, pkl_dir, dmp_level=None, write_json=False,
                 normalization_level=3):
        for target in sorted(self.targets):
            corpus_info = corpus_generator(target, pkl_dir, dmp_level=dmp_level,
                                           write_json=write_json,
                                           normalization_level=normalization_level)
            if corpus_info:
                file_name, corpus_ctr, corpus_data, corpus_voca = corpus_info
                self.num_corpus += corpus_ctr
                self.corpus.append(corpus_data)
                self.vocas += corpus_voca

    def run_fast(self, num_workers=4):
        """
        Corpus generation with multiprcessing support
        """
        # Note that a large file can consume a lot of memory,
        #  leading a deadlock that cannot take advantage of multiprocessing
        #agent = multiprocessing.cpu_count()
        try:
            pool = Pool(processes=num_workers)
            corpora_info = pool.map(corpus_generator, self.targets)
            pool.terminate()

            # Each corpus represents a binary function
            for file_name, corpus_ctr, corpus_data, corpus_voca in corpora_info:
                self.num_corpus += corpus_ctr
                self.corpus.append(corpus_data)
                self.vocas += corpus_voca
        except:
            print ("[-] Error while loading...")
            pass

    def write_corpus(self, res_file_path):
        # Line format: (bin_name, func_name, normalized_instructions, label)
        with open(res_file_path, "w") as corpus_out:
            for c in self.corpus:
                corpus_out.write(c)

    def write_vocas(self, voca_file_path):
        # Line format: (normalized instruction, occurrence)
        with open(voca_file_path, 'w') as voca_out:
            for voca in sorted(self.vocas.keys()):
                voca_out.write("%s, %d\n" % (voca, self.vocas[voca]))



def training_corpus_generator(binary_dir, corpus_dir, pkl_dir, num_workers=None):
    """
    Generating corpus for pretraining
    """
    DG = CorpusGenerator(binary_dir)

    if num_workers:
        DG.run_fast(num_workers=num_workers)
    else:
        DG.run_slow(pkl_dir, dmp_level=None, write_json=False, normalization_level=3)

    binary_dir = binary_dir[:-1] if binary_dir.endswith("/") else binary_dir
    corpus_fp = os.path.join(corpus_dir, 'pretrain.' +\
                             os.path.basename(binary_dir) + '.corpus.txt')
    voca_fp = os.path.join(corpus_dir, 'pretrain.' +\
                           os.path.basename(binary_dir) + '.corpus.voca.txt')

    DG.write_corpus(corpus_fp)
    logging.info ("[+] Generated corpus: %s" % (corpus_fp))
    DG.write_vocas(voca_fp)
    logging.info ("[+] Generated corpus voca: %s" % (voca_fp))



def binsim_corpus_generator(binary_path, result_corpus_fp, pkl_dir):

    corpus = {}
    corpus2 = {}
    label_pos = {'clangO0': 0, 'clangO1': 1, 'clangO2': 2, 'clangO3': 3,
                 'gccO0': 4, 'gccO1': 5, 'gccO2': 6, 'gccO3': 7}

    pos_label = {0: 'clangO0', 1: 'clangO1', 2: 'clangO2', 3: 'clangO3',
        4: 'gccO0', 5: 'gccO1', 6: 'gccO2', 7: 'gccO3' }
    '''

    label_pos = { 'gccO0': 0, 'gccO1': 1, 'gccO2': 2, 'gccO3': 3}

    pos_label = {0: 'gccO0', 1: 'gccO1', 2: 'gccO2', 3: 'gccO3' }

    '''
    # Generate a binsim corpus from pickle files (slow)
    if os.path.isdir(binary_path):
        DG = CorpusGenerator(binary_path)
        bar = tqdm.tqdm(sorted(DG.targets),
                        desc="Corpus loading",
                        total=len(DG.targets),
                        bar_format="{l_bar}{r_bar}")
        for target in bar:
            file_name = os.path.basename(target)
            pkl_dmp_path = os.path.join(pkl_dir, file_name + ".pkl")
            if not os.path.exists(pkl_dmp_path):
                continue

            BS = pickle.load(open(pkl_dmp_path, 'rb'))
            bin_name = BS.bin_name.split("-")[0]
            label = BS.compiler + BS.opt_level

            for fs in BS.fns_summaries:
                identifier = bin_name + '@' + fs.fn_name
                try:
                    if fs.is_linker_func:
                        continue

                    # sometimes a function may not be appeared in all optlevels
                    if identifier not in corpus:
                        corpus[identifier] = [('', '','')] * len(label_pos)

                    normalized_instrs = [x for x in filter(None, fs.normalized_instrs)]

                    corpus[identifier][label_pos[label]] = ','.join(normalized_instrs)
                    
                except TypeError:
                    pass

    with open(result_corpus_fp, "w") as f:
        # FORMAT
        #   [fn1_normalized_instrs, fn2_normalized_instrs, ground_truth, label]
        # Generate positive pairs of binary functions
        funcs = sorted(corpus.keys())
        
        num_paired_corpus = 0
        same_nis_ctr = 0
        
        bar1 = tqdm.tqdm(enumerate(funcs),
                        desc="Generating positive samples",
                        total=len(funcs),
                        bar_format="{l_bar}{r_bar}")
        for idx, func_name in bar1:
            binary_funcs = corpus[func_name]
            for func1, func2 in combinations(range(len(binary_funcs)), 2):
                nis1, nis2 = binary_funcs[func1], binary_funcs[func2]
                # sometimes a function may not be appeared in all optlevels
                if len(nis1) == 0 or len(nis2) == 0\
                              or not(isinstance(nis1, str) and isinstance(nis2, str)):
                    continue
                # too short functions are not our target
                if 5 < len(nis1.split(','))\
                        and 5 < len(nis2.split(',')):
                    if nis1 == nis2:
                        same_nis_ctr += 1
                    gt = func_name + ':' + pos_label[func1] + ":" + pos_label[func2]
                    f.write("%s\t%s\t%s\t%d\n" % (nis1, nis2, gt, 1))
                    num_paired_corpus += 1 
                    
        print("# of total pairs from the corpus: %d" % num_paired_corpus)
        print("# of identical NIS pairs from the corpus: %d" % same_nis_ctr)

        # Generate different pairs of binary functions
        bar2 = util.ProgressBar(num_paired_corpus, name="Negative sampling: %d" % num_paired_corpus)
        num_diff_corpus = 0
        num_fns = len(corpus)

        while num_diff_corpus < num_paired_corpus:
            x = random.randint(0, num_fns // 2 - 1)
            y = random.randint(num_fns // 2, num_fns - 1)
            f1_name, f2_name = funcs[x], funcs[y]
            any_opt1, any_opt2 = random.randint(0, 7), random.randint(0, 7)
            nis1, nis2 = corpus[f1_name][any_opt1], corpus[f2_name][any_opt2]
            if len(nis1) == 0 or len(nis2) == 0\
                          or not(isinstance(nis1, str) and isinstance(nis2, str)):
                continue
            if not(5 < len(nis1.split(','))\
                    and 5 < len(nis2.split(','))):
                continue
            
            gt = f1_name + ':' + pos_label[any_opt1] + ":" + f2_name + ":" + pos_label[any_opt2]
            f.write("%s\t%s\t%s\t%d\n" % (nis1, nis2, gt, 0))

            num_diff_corpus += 1
            bar2 += 1
        bar2.finish()

        print("# of different pairs from the corpus: %d" % num_paired_corpus)



def cve_corpus_generator(corpus_path, result_corpus_fp, pkl_dir):
    corpus = {}
    query = {}

    label_pos = {'clangO0': 0, 'clangO1': 1, 'clangO2': 2, 'clangO3': 3,
                 'gccO0': 4, 'gccO1': 5, 'gccO2': 6, 'gccO3': 7}

    pos_label = {0: 'clangO0', 1: 'clangO1', 2: 'clangO2', 3: 'clangO3',
        4: 'gccO0', 5: 'gccO1', 6: 'gccO2', 7: 'gccO3' }

    # predefined function of interest
    vul_func_name = {
        'openssl': ['tls1_process_heartbeat', 'dtls1_process_heartbeat', 'ssl3_get_new_session_ticket',
                    'OBJ_obj2txt', 'dtls1_get_message_fragment'],
        'ntp': ['crypto_recv', 'ctl_putdata', 'configure'],
        'libav': ['decode_cell_data'],
    }

    addr_to_id = {}

    if os.path.isdir(corpus_path):
        DG = CorpusGenerator(corpus_path)
        bar = tqdm.tqdm(sorted(DG.targets),
                        desc="Corpus loading",
                        total=len(DG.targets),
                        bar_format="{l_bar}{r_bar}")

        query_path = corpus_path[:-1]+"_strip" if corpus_path.endswith("/") else corpus_path+"_strip"
        query_pkl_dir = pkl_dir[:-1]+"_strip" if pkl_dir.endswith("/") else pkl_dir+"_strip"
        DG2 = CorpusGenerator(query_path)
        bar2 = tqdm.tqdm(sorted(DG2.targets),
                        desc="Query loading",
                        total=len(DG2.targets),
                        bar_format="{l_bar}{r_bar}")

        # create database that holds function of interest
        for target in bar:
            # only gcc compiler is used for compiler
            # but to find start_addr of func, also read clang binaries
            file_name = os.path.basename(target)
            prj_name = file_name.split("_")[0]
            pkl_dmp_path = os.path.join(pkl_dir, file_name + ".pkl")

            BS = pickle.load(open(pkl_dmp_path, 'rb'))
            bin_name = BS.bin_name.rsplit("-", 2)[0]
            label = BS.compiler + BS.opt_level

            if prj_name not in corpus:
                corpus[prj_name] = {}
            prj_vul_func_name = vul_func_name[prj_name]

            for fs in BS.fns_summaries:
                identifier = bin_name + '@' + fs.fn_name
                try:
                    if fs.is_linker_func:
                        continue

                    # sometimes a function may not be appeared in all optlevels
                    normalized_instrs = [x for x in filter(None, fs.normalized_instrs)]

                    # create database that holds function of interest
                    if fs.fn_name in prj_vul_func_name:
                        if identifier not in corpus[prj_name]:
                            corpus[prj_name][identifier] = \
                                [('', '','')] * len(label_pos)
                        corpus[prj_name][identifier][label_pos[label]] = \
                            ','.join(normalized_instrs)

                    # to get ground truth in stripped binaries (address is same with unstripped one)
                    strip_identifier = bin_name + '_' + str(label) + '_' + \
                                       hex(fs.fn_start)
                    assert strip_identifier not in addr_to_id
                    addr_to_id[strip_identifier] = identifier

                except TypeError:
                    pass

        # extract query functions from query binary
        for target in bar2:
            file_name = os.path.basename(target)
            # queries are only from clang
            if '-gcc-O' in file_name:
                continue
            prj_name = file_name.split("_")[0]
            pkl_dmp_path = os.path.join(query_pkl_dir, file_name + ".pkl")

            BS = pickle.load(open(pkl_dmp_path, 'rb'))
            bin_name = BS.bin_name.rsplit("-", 2)[0]
            label = BS.compiler + BS.opt_level

            if prj_name not in query:
                query[prj_name] = {}

            for fs in BS.fns_summaries:
                # find ground truth if the function was in the unstripped binary
                strip_identifier = bin_name + '_' + str(label) + '_' + \
                                   hex(fs.fn_start)
                if strip_identifier in addr_to_id:
                    identifier = addr_to_id[strip_identifier]
                else:
                    identifier = bin_name + '@' + fs.fn_name

                try:
                    if fs.is_linker_func:
                        continue

                    # sometimes a function may not be appeared in all optlevels
                    if identifier not in query[prj_name]:
                        query[prj_name][identifier] = \
                            [('', '','')] * len(label_pos)
                    normalized_instrs = [x for x in filter(None, fs.normalized_instrs)]
                    query[prj_name][identifier][label_pos[label]] = \
                        ','.join(normalized_instrs)

                except TypeError:
                    pass

    with open(result_corpus_fp, "w") as f:
        # FORMAT
        #   [fn1_normalized_instrs, fn2_normalized_instrs, gt, label]
        for pn, cp in corpus.items():
            qr = query[pn]
            cp_funcs_id = sorted(cp.keys())
            qr_funcs_id = sorted(qr.keys())
            
            for idx, cp_fn in enumerate(cp_funcs_id):
                cp_funcs = cp[cp_fn]
                # only gcc binaries are in the database
                for cp_func in [4, 5, 6, 7]:
                    cp_nis = cp_funcs[cp_func]
                    if len(cp_nis) == 0 or not(isinstance(cp_nis, str)):
                        continue

                    for idx2, qr_fn in enumerate(qr_funcs_id):
                        qr_funcs = qr[qr_fn]
                        for qr_func in range(len(qr_funcs)):
                            qr_nis = qr_funcs[qr_func]
                            if len(qr_nis) == 0 or not(isinstance(qr_nis, str)):
                                continue

                            label = int(cp_fn == qr_fn)
                            if label:
                                gt = cp_fn + ':' + pos_label[cp_func] + ':' + \
                                     pos_label[qr_func]
                            else:
                                gt = cp_fn + ':' + pos_label[cp_func] + ':' + \
                                     qr_fn + ':' + pos_label[qr_func]

                            f.write("%s\t%s\t%s\t%d\n"
                                    % (cp_nis, qr_nis, gt, label))



def all_fns_generator(binary_path, pkl_dir, cve_flag):
    """
    Write all assembly functions in each binary
        (Line format: (func_id, func_name, normalized_instructions)
    """
    if os.path.isdir(binary_path):
        DG = CorpusGenerator(binary_path)
        bar = tqdm.tqdm(sorted(DG.targets),
                        desc="Corpus loading",
                        total=len(DG.targets),
                        bar_format="{l_bar}{r_bar}")
        for target in bar:
            file_name = os.path.basename(target)
            pkl_dmp_path = os.path.join(pkl_dir, file_name + ".pkl")

            BS = pickle.load(open(pkl_dmp_path, 'rb'))
            #bin_name = BS.bin_name.split("-")[0]
            #bin_name = BS.bin_name
            #label = BS.compiler + BS.opt_level
            all_fns_fp = open(os.path.join(pkl_dir,file_name+'.fns'),'w')
            for fs in BS.fns_summaries:
                try:
                    normalized_inst = [x for x in filter(None, fs.normalized_instrs)]
                    if not cve_flag:
                        all_fns_fp.write(str(fs.fn_idx)+"\t"+fs.fn_name+"\t"+','.join(normalized_inst)+'\n')
                    else:
                        all_fns_fp.write(str(fs.fn_idx)+"\t"+fs.fn_name+"\t"+hex(fs.fn_start)+"\t"+','.join(normalized_inst)+'\n')
                except TypeError:
                    pass



class CorpusDataset(Dataset):
    def __init__(self, corpus_path, encoding="utf-8"):
        self.num_data = 0
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.corpus = []

        with open(corpus_path, "r", encoding=encoding) as f:
            for line in tqdm.tqdm(f, desc="[+] Loading Dataset",
                                  total=self.num_data):
                corpus = line.strip()
                self.corpus.append(corpus)
            self.num_data = len(self.corpus)

    def __len__(self):
        return self.num_data

    def __getitem__(self, item):
        return self.corpus[item]

def corpus_split(corpus_fp, type="pretrained",
                 ratio=(0.9, 0.05, 0.05)):
    assert sum(ratio) == 1.0, "Ratio sum must be 1.0!!"
    torch.manual_seed(99)
    dataset = CorpusDataset(corpus_fp)

    train_ratio, valid_ratio, test_ratio = ratio
    train_ctr, valid_ctr, test_ctr = 0, 0, 0
    train_len = train_ratio * len(dataset)
    valid_len = valid_ratio * len(dataset)
    test_len = len(dataset) - train_len - valid_len

    print("[+] Split the given dataset (%s): "
          "training (%d), validation (%d), test (%d)" \
          % (ratio, train_len, valid_len, test_len))

    print("[+] Store each corpus into separate files")
    dir_path = os.path.dirname(corpus_fp)
    common = os.path.basename(corpus_fp).split('.')[0] \
        if type=="pretrained" else "binsim." + os.path.basename(corpus_fp).split('.')[1]

    train_fp = os.path.join(dir_path, common + ".train.corpus.txt")
    valid_fp = os.path.join(dir_path, common + ".valid.corpus.txt")
    test_fp = os.path.join(dir_path, common + ".test.corpus.txt")

    with open(train_fp, "w") as f, open(valid_fp, "w") as g, open(test_fp, "w") as h:
        for data in DataLoader(dataset, shuffle=True):
            if type == "pretrained":
                file_name, func_name, nf, label = data[0].split("\t")
                write_format = "%s\t%s\t%s\t%s\n" \
                               % (file_name, func_name, nf, label)
            elif type == "binsimtask":
                nf1, nf2, gt, label = data[0].split("\t")
                write_format = "%s\t%s\t%s\t%s\n" \
                               % (nf1, nf2, gt, label)
            else:
                print ("[-] Unknown type: only supports "
                       "[pretrained, binsimtask] for now!")
                sys.exit(1)

            # Split different data into different files
            if train_ctr < train_len:
                train_ctr += 1
                f.write(write_format)
            elif valid_ctr < valid_len:
                valid_ctr += 1
                g.write(write_format)
            elif test_ctr < test_len:
                test_ctr += 1
                h.write(write_format)



if __name__ == '__main__':

    usage = "Usage: [-d <binary-dir> -pkl <pickle-dir> | -t | -b] (Use -h for help)"
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("-d", "--binary_dir", required=False, type=str,
                        help="binary directory for corpus")
    parser.add_argument("-f", "--target_file", required=False, type=str, 
                        help="Target binary")
    parser.add_argument("-pkl", "--pkl_dir", required=False, type=str,
                        help="pkl directory for corpus")
    parser.add_argument("-o", "--corpus_dir", required=False, type=str,
                        help="corpus directory (output directory)")

    # Options pertaining to corpus generation
    parser.add_argument("-t", "--training_gen", dest="train", action="store_true",
                        help="Generating training corpus")
    parser.add_argument("-b", "--binsim_gen", dest="binsim", action="store_true",
                        help="Generating downstream corpus")
    parser.add_argument("-fns", "--all_fns_gen", dest="all_fns_gen", action="store_true",
                        help="Create assembly code for all functions")
    parser.add_argument("-c", "--cve", dest="cve", action="store_true",
                        help="Create corpus for realistic scenario")
    parser.add_argument("-w", "--num_workers", type=int,
                        help="Number of workers for multiprocessing")

    # Options pertaining to dumping binary information
    parser.add_argument("-j", "--json_dump", dest="json_dump", action="store_true",
                      help="Dump binary info in a json format (with -f)")
    parser.add_argument("-x", "--txt_dump", dest="txt_dump", action="store_true",
                      help="Dump binary info in a txt format (with -f)")
    parser.add_argument("-l", "--dump_level", dest="dmp_level", type=int,
                        help="Dump level for a txt format (with -f/-x)"
                        "0=fun, 1=fun(detail), 2=bbl, 3=ins")
    parser.add_argument("-n", "--norm_level", dest="norm_level", type=int, default=3,
                        help="Normalization level for a txt format (with -j/-x)"
                             "1=immvals, 2=regs, 3=ptrs")

    # Corpus splitting
    parser.add_argument("-p", "--split_data", dest="split", action="store_true",
                        help="Split data into training/validation/test (with -f)")
    parser.add_argument("-y", "--data_type", dest="data_type", type=str,
                        default="pretrained",
                        help="Support type for splitting: pretrained, binsimtask" 
                             "with (-p)")

    args = parser.parse_args()

    # Generate *.fns files in pkl_dir
    # *.fns files contain all assembly functions in each binary
    if args.all_fns_gen:
        all_fns_generator(args.binary_dir, args.pkl_dir, args.cve)

    # Generating a pair of (*.corpus.txt, *.corpus.voca.txt) for pretraining
    # Note that num_workers should be given for multiprocessing
    elif args.train and os.path.exists(args.corpus_dir) and\
            os.path.exists(args.binary_dir) and os.path.exists(args.pkl_dir):
        corpus_dir = args.corpus_dir[:-1] if args.corpus_dir.endswith("/") else args.corpus_dir
        binary_dir = args.binary_dir[:-1] if args.binary_dir.endswith("/") else args.binary_dir
        pkl_dir = args.pkl_dir+'/' if not args.binary_dir.endswith("/") else args.pkl_dir
        training_corpus_generator(binary_dir, corpus_dir, args.pkl_dir, num_workers=args.num_workers)

    # Generating a corpus for the downstream task: binary similarity
    # Line format (fn1_normalized_instrs, fn2_normalized_instrs,
    #              ground_truth, label)
    elif args.binsim and os.path.exists(args.corpus_dir) and\
            os.path.exists(args.binary_dir) and os.path.exists(args.pkl_dir):
        corpus_dir = args.corpus_dir[:-1] if args.corpus_dir.endswith("/") else args.corpus_dir
        binary_dir = args.binary_dir[:-1] if args.binary_dir.endswith("/") else args.binary_dir
        pkl_dir = args.pkl_dir+'/' if not args.binary_dir.endswith("/") else args.pkl_dir
        binsim_corpus_fp = corpus_dir + "/binsim." + os.path.basename(binary_dir) + ".corpus.txt"
        binsim_corpus_generator(binary_dir, binsim_corpus_fp, pkl_dir)
        #binsim_corpus_generator('cve-only.csv', 'binsim.cve.corpus.txt')
        #binsim_cve_corpus_generator('cve-only.csv', 'binsim.cve.corpus.txt')

    # Generating a corpus for realistic scenario
    # Compare all functions in query binaries with functions of interest (i.e., vulnerable functions)
    elif args.cve and os.path.exists(args.corpus_dir) and\
            os.path.exists(args.binary_dir) and os.path.exists(args.pkl_dir):
        binary_dir = args.binary_dir[:-1] if args.binary_dir.endswith("/") else args.binary_dir
        pkl_dir = args.pkl_dir+'/' if not args.binary_dir.endswith("/") else args.pkl_dir
        binsim_corpus_fp = args.corpus_dir + "/cve." + os.path.basename(args.binary_dir) + "corpus.txt"
        cve_corpus_generator(binary_dir, binsim_corpus_fp, pkl_dir)



    # python corpusgen.py -f ./test/findutils_find-amd64-clang-O0 -j
    elif args.json_dump and os.path.isfile(args.target_file):
        nn, _, _, _ = run_normalization(args.target_file,
                                        normalization_level=args.norm_level)
        json_fp = args.target_file + ".json"
        json_corpus = nn.json_dump(json_fp)

    # python corpusgen.py -f ./test/findutils_find-amd64-clang-O0 -x -l 0 -n 3
    elif args.txt_dump and os.path.isfile(args.target_file):
        res_fp = args.target_file + ".info.txt"
        nn, _, _, _ = run_normalization(args.target_file,
                                        normalization_level=args.norm_level)
        nn.write_bin_info(res_fp, level=args.dmplevel, resolve_callee=True)

    # python3 corpusgen.py -f ./data/binsim.testsuite-all.corpus.txt -y binsimtask -p
    elif args.split and os.path.isfile(args.target_file):
        corpus_fp = args.target_file
        corpus_split(corpus_fp, type=args.data_type, ratio=(0.9, 0.05, 0.05))

    else:
        parser.print_help()

        

