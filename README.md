# BinShot
This is the official repository for BinShot (ACSAC 22'),
which is a practical binary code similarity detection tool with BERT-based transferable similarity learning.

## Requirements
```
pip install -r requirements.txt
```
Besides, you need to install the followings: 
* python3 (tested on 3.8)
* IDA Pro (tested on 7.6)
* pytorch (tested on 1.11)



## Run codes with published data
### Download published data
The proprocessed data used in our paper can be found by following google drive link:   
https://bit.ly/3DlkFJk   
Download the data, and then move them into "corpus" directory.


### Pretraining
```
python3 bert_mlm.py \
            -cd corpus/pretrain.all.corpus.txt \
            -vp corpus/pretrain.all.corpus.voca \
            -op models/pretrain
```

### Finetuning & Evaluation
```
python3 binshot.py \
            -bm models/pretrain/model_bert/bert_ep19.model \
            -vp corpus/pretrain.all.corpus.voca \
            -op models/downstream \
            -r all \
            -tn corpus/binsim.all.train.corpus.txt \
            -vd corpus/binsim.all.valid.corpus.txt \
            -tt corpus/binsim.all.test.corpus.txt
```
* If you want to get the metrics per compiler-optimization level pair(i.e., clangO0 - gccO2),
```
python result.py -s models/downstream/pred.test.all_all -v models/downstream
```

### Transferability Evaluation
```
python3 binshot.py \
            -bm models/pretrain/model_bert/bert_ep19.model \
            -vp corpus/pretrain.all.corpus.voca \
            -op models/spec06 \
            -r spec06 \
            -tn corpus/binsim.spec06.train.corpus.txt \
            -vd corpus/binsim.spec06.valid.corpus.txt \
            -tt corpus/binsim.spec06.test.corpus.txt

python3 binshot.py \
            -bm models/pretrain/model_bert/bert_ep19.model \
            -vp corpus/pretrain.all.corpus.voca \
            -op models/spec17 \
            -r spec17 \
            -tn corpus/binsim.spec17.train.corpus.txt \
            -vd corpus/binsim.spec17.valid.corpus.txt \
            -tt corpus/binsim.spec17.test.corpus.txt

python3 binshot.py \
            -bm models/[spec06,spec17]/model_sim/bert_ep19.model \
            -fm models/[spec06,spec17]/model_sim/sim_ep19.model \
            -vp corpus/pretrain.all.corpus.voca \
            -op models/[spec06,spec17] \
            -r [gnu,spec06,spec17,rwp] \
            -tt corpus/binsim.[gnu,spec06,spec17,rwp].test.corpus.txt
```

### Practicality Evaluation
```
python3 binshot.py \
            -bm models/downstream/model_sim/bert_ep19.model \
            -fm models/downstream/model_sim/sim_ep19.model \
            -vp corpus/pretrain.all.corpus.voca \
            -op models/downstream \
            -r cve \
            -tt corpus/cve.corpus.txt
```
* In this evaluation, we need to compare the each functions in compiler-optimization level, not each pairs.
```
python result_cve.py -s models/downstream/pred.test.cve_all -v models/downstream -t 0.5 -f
```



## Generate input files with your own binaries
The following codes will be run with sample binaries in our repo.

### Advance preparation
* Binary should have execution permission.
* Binary name format should be "binname-IA-compiler-optlv" (e.g., find-amd64-gcc-O2)
* run following commands
```
mkdir -p norm/findutils norm/cve norm/cve_strip
```

### Running IDA Pro
```
bash gen_ida.sh binary/findutils/
bash gen_ida.sh binary/cve/
bash gen_ida.sh binary/cve_strip/
```

### Normalizing assembly codes
```
bash gen_norm.sh binary/findutils/ norm/findutils/
bash gen_norm.sh binary/cve/ norm/cve/
bash gen_norm.sh binary/cve_strip/ norm/cve_strip/
```

### Corpus generation for pretraining
```
python3 corpusgen.py -d binary/findutils/ -pkl norm/findutils/ -o corpus/ -t
python3 voca.py corpus/pretrain.findutils.corpus.voca.txt
```

### Corpus generation for finetuning
```
python3 corpusgen.py -d binary/findutils/ -pkl norm/findutils/ -o corpus/ -b
python3 corpusgen.py -f corpus/binsim.findutils.corpus.txt -y binsimtask -p
```

### Corpus generation for a realistic scenario
```
python3 corpusgen.py -d binary/cve/ -pkl norm/cve/ -o corpus/ -c
```



## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If your research employs BinShot, please cite the following paper:
```
@INPROCEEDINGS{binshot,   
  author = {Sunwoo Ahn and Seonggwan Ahn and Hyungjoon Koo and Yunheung Paek},   
  title = {Practical Binary Code Similarity Detection with BERT-based
		   Transferable Similarity Learning}   
  booktitle = {Proceedings of the 38th Annual Computer Security
               Applications Conference (ACSAC)},   
  month = {Dec.},   
  year = {2022},   
  location = {Austin, Texas}   
}
```

