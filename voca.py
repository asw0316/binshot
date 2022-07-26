import pickle,sys, os

class WordVocab(object):
    def __init__(self, voca_list):
        super(WordVocab, self).__init__()
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        self.voca_list = voca_list
        self.special_symbol_idx = {
                        '<pad>': self.pad_index,
                        '<unk>': self.unk_index,
                        '<eos>': self.eos_index,
                        '<sos>': self.sos_index,
                        '<mask>': self.mask_index
                    }

        for voca in self.voca_list:
            if voca not in self.special_symbol_idx:
                self.special_symbol_idx[voca] = len(self.special_symbol_idx)

        self.idx_special_sym = dict((idx, char) for char, idx in self.special_symbol_idx.items())
        print('vocab size: %d' %self.vocab_size)

    def voca_idx(self, vocas):
        if isinstance(vocas, list):
            return [self.special_symbol_idx.get(voca, self.unk_index) \
                    for voca in vocas]
        else:
            return self.special_symbol_idx.get(vocas, self.unk_index)

    def idx_voca(self, idxs):
        if isinstance(idxs, list):
            return [self.idx_special_sym.get(i) for i in idxs]
        else:
            return self.idx_special_sym.get(idxs)

    @property
    def vocab_size(self):
        return len(self.special_symbol_idx)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

def build_voca(voca_fp, out_fp):
    pad = '<pad>'
    unk = '<unk>'
    eos = '<eos>'
    sos = '<sos>'
    mask = '<mask>'

    instructions = []
    with open(voca_fp, 'r') as f:
        for line in f.readlines():
            word = line.split(',')[0].strip()
            instructions.append(word)

    vocas = [pad, unk, eos, sos, mask] + instructions

    wv = WordVocab(vocas)
    wv.save_vocab(out_fp)
    print("[+] Saved all %d (normalized) instructions"
          " as a full vocabulary set into %s" \
          % (wv.vocab_size, out_fp))


if __name__ == '__main__':
    f = sys.argv[1]
    out_f = f[:-4]
    build_voca(f, out_f)

