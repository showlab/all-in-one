> The dataset preparation is the same as All-in-one-B.

## All-in-one-B+ [2 Dataset]
- Video: Webvid
- Image: CC3M

```bash
cd CoTraining

python run.py with data_root=DataSet num_gpus=8 num_nodes=1 \
num_frames=3 \
task_mlm_vtm_cotrain whole_word_masking=True step200k per_gpu_batchsize=4 backend='v100'

```

## All-in-one-B+ [7 Dataset]
- Video: Webvid, YTTemporal, HowTo100M
- Image: CC3M

```bash
cd CoTraining

python run.py with data_root=DataSet num_gpus=8 num_nodes=1 \
num_frames=3 \
task_mlm_vtm_cotrain_seven whole_word_masking=True step200k per_gpu_batchsize=4 backend='v100'
```