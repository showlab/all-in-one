# Evaluation

## VQA

### Evaluate TGIF

#### TGIF-QA FrameQA

```bash
python run.py with data_root=DataSet num_gpus=8 \
num_nodes=1 \
num_frames=3 \
per_gpu_batchsize=8 task_finetune_tgifqa \
load_path="pretrained/all-in-one-base.ckpt"
```

#### TGIF-QA Action/Transition

Modify line 19 in [`tgifqa`](AllInOne/datasets/tgifqa.py) for transition/action.

```bash
python run.py with data_root=DataSet num_gpus=8 \
num_nodes=1 \
num_frames=3 \
per_gpu_batchsize=16 task_finetune_tgif_action_trans \
load_path="pretrained/all-in-one-base.ckpt"
```


### MSRVTT-QA

```bash
python run.py with data_root=DataSet num_gpus=8 \
num_nodes=1 \
num_frames=3 \
per_gpu_batchsize=16 task_finetune_msrvttqa \
load_path="pretrained/all-in-one-base.ckpt"
```


### MSVD-QA
```bash
python run.py with data_root=DataSet num_gpus=8 \
num_nodes=1 \
num_frames=3 \
per_gpu_batchsize=16 task_finetune_msvdqa \
load_path="pretrained/all-in-one-base.ckpt"
```


## Action Recognition (Linear Evaluation)

### K400
```bash
python run.py \
with data_root=DataSet num_gpus=8 num_nodes=1 \
per_gpu_batchsize=16 task_finetune_action_recognition_k400 \
num_frames=8 linear_evaluation=True \
load_path="pretrained/all-in-one-base.ckpt"
```

[comment]: <> (|  Accuracy   | Report in Paper  | Trained Log |)

[comment]: <> (|  ----  | ----  | --- |)

[comment]: <> (| 52.3  | 50.8 | [anonymous]&#40;anonymous&#41; |)

### HMDB51
```bash
python run.py \
with data_root=DataSet num_gpus=8 num_nodes=1 \
per_gpu_batchsize=8 task_finetune_action_recognition_hmdb51 \
num_frames=3 linear_evaluation=True backend='a100' \
load_path="pretrained/all-in-one-base.ckpt"
```


[comment]: <> (|  Accuracy   | Report in Paper  | Trained Log |)

[comment]: <> (|  ----  | ----  | --- |)

[comment]: <> (| 51.2 | 50.8 | [anonymous]&#40;anonymous&#41; |)


## Zero-shot Multiple-choice

### LSMDC

```bash
python run.py with data_root=DataSet num_gpus=8 num_nodes=1 \
per_gpu_batchsize=8 task_finetune_lsmdcchoice test_only=True \
num_frames=3 \
load_path="pretrained/all-in-one-base.ckpt"
```

[comment]: <> (|  Accuracy   | Report in Paper  | Trained Log |)

[comment]: <> (|  ----  | ----  | --- |)

[comment]: <> (| 56.5 | 56.3 | [anonymous]&#40;anonymous&#41; |)


### MSRVTT

```bash
python run.py with data_root=DataSet num_gpus=8 num_nodes=1 \
per_gpu_batchsize=8 task_finetune_msrvttchoice test_only=True \
num_frames=3 \
load_path="pretrained/all-in-one-base.ckpt"
```

[comment]: <> (|  Accuracy   | Report in Paper  | Trained Log |)

[comment]: <> (|  ----  | ----  | --- |)

[comment]: <> (| 79.6 | 80.3 | [anonymous]&#40;anonymous&#41; |)

## Retrieval

### MSRVTT
#### VTC only
```bash
python run.py with \
data_root=DataSet num_gpus=8 num_nodes=1 \
per_gpu_batchsize=32 task_finetune_only_ind_itc_msrvtt_randaug \
num_frames=3 \
load_path="pretrained/all-in-one-base.ckpt"
```

#### VTC + VTM

```bash
python run.py with \
data_root=DataSet num_gpus=8 num_nodes=1 \
per_gpu_batchsize=6 task_finetune_ind_itc_irtr_msrvtt_randaug \
num_frames=3 \
load_path="pretrained/all-in-one-base.ckpt"
```