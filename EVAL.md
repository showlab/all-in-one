# Evaluation

## 1. VQA

### Evaluate TGIF

#### TGIF-QA FrameQA

```bash
python run.py with data_root=DataSet num_gpus=8 \
num_nodes=1 \
num_frames=3 \
per_gpu_batchsize=8 task_finetune_tgifqa \
load_path="pretrained/all-in-one-base.ckpt"
```

|  Accuracy   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 64.3  | 64.0 | [anonymous](anonymous) |

#### TGIF-QA Action/Transition

Modify line 19 in [`tgifqa`](AllInOne/datasets/tgifqa.py) for transition/action.

```bash
python run.py with data_root=DataSet num_gpus=8 \
num_nodes=1 \
num_frames=3 \
per_gpu_batchsize=16 task_finetune_tgif_action_trans \
load_path="pretrained/all-in-one-base.ckpt"
```

|  Accuracy   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 93.0  | 92.5 | [google driver](https://drive.google.com/file/d/1GQLvIKpEC_flfOFx9GA7c7Ks26cfcvcK/view?usp=sharing) |


### MSRVTT-QA

```bash
python run.py with data_root=DataSet num_gpus=8 \
num_nodes=1 \
num_frames=3 \
per_gpu_batchsize=16 task_finetune_msrvttqa \
load_path="pretrained/all-in-one-base.ckpt"
```

|  Accuracy   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 42.9  | 42.5 | [anonymous](anonymous) |

### MSVD-QA
```bash
python run.py with data_root=DataSet num_gpus=8 \
num_nodes=1 \
num_frames=3 \
per_gpu_batchsize=16 task_finetune_msvdqa \
load_path="pretrained/all-in-one-base.ckpt"
```

|  Accuracy   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 46.1  | 46.5 | [anonymous](anonymous) |


## 2. Action Recognition (Linear Evaluation)

### K400
```bash
python run.py \
with data_root=DataSet num_gpus=8 num_nodes=1 \
per_gpu_batchsize=16 task_finetune_action_recognition_k400 \
num_frames=8 linear_evaluation=True \
load_path="pretrained/all-in-one-base.ckpt"
```

|Accuracy|Report in Paper| Trained Log |
| ---- |---- | --- |
| 52.3  | 50.8 | [anonymous](anonymous) |

### HMDB51
```bash
python run.py \
with data_root=DataSet num_gpus=8 num_nodes=1 \
per_gpu_batchsize=8 task_finetune_action_recognition_hmdb51 \
num_frames=3 linear_evaluation=True backend='a100' \
load_path="pretrained/all-in-one-base.ckpt"
```


|  Accuracy   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 51.2 | 50.8 | [anonymous](anonymous) |


## 3. Zero-shot Multiple-choice

### LSMDC

```bash
python run.py with data_root=DataSet num_gpus=8 num_nodes=1 \
per_gpu_batchsize=8 task_finetune_lsmdcchoice test_only=True \
num_frames=3 \
load_path="pretrained/all-in-one-base.ckpt"
```

|  Accuracy   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 56.5 | 56.3 | [anonymous](anonymous) |


### MSRVTT

```bash
python run.py with data_root=DataSet num_gpus=8 num_nodes=1 \
per_gpu_batchsize=8 task_finetune_msrvttchoice test_only=True \
num_frames=3 \
load_path="pretrained/all-in-one-base.ckpt"
```

|  Accuracy   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 79.6 | 80.3 | [anonymous](anonymous) |

## 4. Retrieval

### MSRVTT
#### VTC only
```bash
python run.py with \
data_root=DataSet num_gpus=8 num_nodes=1 \
per_gpu_batchsize=32 task_finetune_only_ind_itc_msrvtt_randaug \
num_frames=3 \
load_path="pretrained/all-in-one-base.ckpt"
```

|  R1   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 35.4 | 35.7 | [anonymous](anonymous) |

#### VTC + VTM

```bash
python run.py with \
data_root=DataSet num_gpus=8 num_nodes=1 \
per_gpu_batchsize=6 task_finetune_ind_itc_irtr_msrvtt_randaug \
num_frames=3 \
load_path="pretrained/all-in-one-base.ckpt"
```

|  R1   | Report in Paper  | Trained Log |
|  ----  | ----  | --- |
| 36.7 | 37.1 | [anonymous](anonymous) |