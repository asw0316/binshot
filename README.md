# BinShot
This is the official repository for BinShot, which is a practical binary code similarity detection tool with BERT-based transferable similarity learning.

## Requirements
```
pip install -r requirements.txt
```
Besides, you need to install the followings: 
* python3 (tested on 3.8)
* IDA Pro (tested on 7.6)
* pytorch (tested on 1.11)
* tensorflow (tested on 2.2.0)
* tensorboard (tested on 2.2.2)

## Run codes with example binaries
### Advance preparation
* Binary should have execution permission.
* Binary name format should be "binname-IA-compiler-optlv" (e.g., find-amd64-gcc-O2)
* run following commands
```
mkdir norm corpus models
mkdir norm/findutils norm/cve norm/cve_strip
```

### Run IDA
```
bash gen_ida.sh binary/findutils/
```
* trouble shooting: If dmp.gz files are not generated, add the following code to line 9 in ida.py.
```
sys.path.insert(0, '/absolute/path/to/binshot/codes')
```

### Normalize assembly
```
bash gen_norm.sh binary/findutils/ norm/findutils/
```

### Pretraining corpus generation
```
python3 corpusgen.py -d binary/findutils/ -pkl norm/findutils/ -o corpus/ -t
python3 voca.py corpus/pretrain.findutils.corpus.voca.txt
```

### Finetuning corpus generation
```
python3 corpusgen.py -d binary/findutils/ -pkl norm/findutils/ -o corpus/ -b
python3 corpusgen.py -f corpus/binsim.findutils.corpus.txt -y binsimtask -p
```

### Generating copurs for realistic scenario
```
python3 corpusgen.py -d binary/findutils/ -pkl norm/findutils/ -o corpus/ -c
python3 corpusgen.py -f corpus/cve.cve.corpus.txt -y binsimtask -p
```

### Pretraining
```
python3 bert_mlm.py \
            -cd corpus/pretrain.findutils.corpus.txt \
            -vp corpus/pretrain.findutils.corpus.voca \
            -op models/pretrain
```

### Finetuning & evaluation
```
python3 binshot.py \
            -bm models/pretrain/model_bert/bert_ep19.model \
            -vp corpus/pretrain.findutils.corpus.voca \
            -op models/downstream \
            -r findutils \
            -tn corpus/binsim.findutils.train.corpus.txt \
            -vd corpus/binsim.findutils.valid.corpus.txt \
            -tt corpus/binsim.findutils.test.corpus.txt

python3 binshot.py \
            -bm models/downstream/model_sim/bert_ep19.model \
            -fm models/downstream/model_sim/sim_ep19.model \
            -vp corpus/pretrain.findutils.corpus.voca \
            -op models/downstream \
            -r cve \
            -tt corpus/cve.cve.corpus.txt
```

