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

Youtube-dl is slow recently, you may use [yt-dlp](https://github.com/yt-dlp/yt-dlp) as instead.


### Download Dataset Scripts
We provide scripts for download WebVid, CC3M and YT-Temporal in
[Google Driver](https://drive.google.com/drive/folders/12uizpMbjX1Uw7XA5asBy6xbHC-RAVrmd?usp=sharing).



### WebVid [5T]
Download results_2M_train.csv and results_2M_val.csv from [webvid](https://github.com/m-bain/webvid), and then use provided scripts to download webvid.

### HowTo100M [~20T]
Download source video from [howto100m](https://www.di.ens.fr/willow/research/howto100m/).

### YT-Temporal 180M [60T]

Please email to [rowanz](https://github.com/rowanz/merlot/tree/main/data) to access this dataset and then download with our provided scripts.

Notice that this dataset don't provide clean ASR. 
Please follow [merlot](https://github.com/rowanz/merlot/blob/main/data/process.py) to clean the ASR text, we have implement this in yttemporal.py.


## 3. Download Finetune Dataset


### TGIF-QA [134G]
Download raw frames from [Google Driver](https://drive.google.com/file/d/11wdvsTYIPcSTRMVry1tufILiNE4aAMp5/view?usp=sharing).

### MSRVTT [<10G]

```bash
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -P data; unzip data/MSRVTT.zip
```
Please refer to [Frozen](https://github.com/m-bain/frozen-in-time) for more details if you have difficult in downloading this dataset.

### MSVD [1.86G]

```bash
wget -c https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar
```
Please refer to [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) for more details if you have difficult in downloading this dataset.

### K400 [260G]
Download csv file from [here](https://deepmind.com/research/open-source/kinetics).
Then download source video the same as Webvid.

### HMDB51 [<10G]

Download source video from [here](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

### Ego4d [900G]

### LSMDC

### ActivityNet [200G]

Download source video from [google driver](https://drive.google.com/file/d/12YOTnPc4zCwum_R9CSpZAI9ppAei8KMG/view?usp=sharing).

### DiDeMo


## 4. Soft Link and Meta Data

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
Place all annotation file download from google driver in metadata, as shown in below:

![Datasets](figures/metadata.png)

The example of Webvid is as below:
![Datasets](figures/webvid.png)