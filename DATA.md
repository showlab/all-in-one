# Dataset Preparation
We utilize three datasets for pretrain: WebVid2.5M, HowTo100M, YT-Temporal180M.

For downstream datasets, we use 11 datasets in total: TGIF, MSVD, MSRVTT, TVQA, Ego4D, ActivityNet, LSMDC, DiDeMo, Kinetics, HMDB51, VCR.

We train models on these raw videos directly, instead of using off-line extracted feature.
We do not distribute datasets because of the license issue.
Please download these datasets by yourself with following instructions:


## 1. Download Annotations
For simple, we arrange the annotation file from 15 datasets. Include:
activitynet, cc3m, didemo, ego4d, hmdb51, howto100m, k400, lsmdc, msrvtt, msvd, tgif, tvqa, vcr1annots, webvid, yttemporal.


Please download annotation files arranged by us from [google driver](https://drive.google.com/drive/folders/1nXdcVzvA8CoeShk6PhzM8bd0uzCfARXI?usp=sharing).

## 2. Download Pretrain Dataset
All these datasets contain video mainly from YouTube, please install
```python
pip install youtube-dl
```

### Download Dataset Scripts
We provide scripts for download WebVid, CC3M and YT-Temporal in
[Google Driver](https://drive.google.com/drive/folders/12uizpMbjX1Uw7XA5asBy6xbHC-RAVrmd?usp=sharing).



### WebVid
Download results_2M_train.csv and results_2M_val.csv from [webvid](https://github.com/m-bain/webvid), and then use provided scripts to download webvid.

### HowTo100M
Download source video from [howto100m](https://www.di.ens.fr/willow/research/howto100m/).

### YT-Temporal 180M

Please email to [rowanz](https://github.com/rowanz/merlot/tree/main/data) to access this dataset and then download with our provided scripts.

Notice that this dataset don't provide clean ASR. 
Please follow [merlot](https://github.com/rowanz/merlot/blob/main/data/process.py) to clean the ASR text, we have implement this in yttemporal.py.


## 3. Download Finetune Dataset


### TGIF-QA

### MSRVTT

### MSVD

### K400

### HMDB51

### Ego4d

### LSMDC

### ActivityNet

### DiDeMo


## Soft Link and Meta Data

After downloading all these datasets, please prepare these datasets as follow:

### Add soft link
```bash
mkdir dataset
ln -s [path_to_original_dataset] dataset/[lowercase_short_name]
```
As shown in below:

![Datasets](figures/dataset.png)

### Meta data
```bash
mkdir metadata
```
Place all annotation file in metadata, as shown in below:

![Datasets](figures/metadata.png)

The example of Webvid is as below:
![Datasets](figures/webvid.png)