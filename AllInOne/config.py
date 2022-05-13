from sacred import Experiment

ex = Experiment("AllInOne")


def _loss_names(d):
    ret = {
        # pretrain
        "itm": 0,
        "itc": 0,
        "mlm": 0,
        "mpp": 0,
        "ind_itc": 0,
        "vcop": 0,
        # downstream
        "vqa": 0,
        "openend_vqa": 0,
        "mc_vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "multiple_choice": 0,
        'vcr_q2a': 0,
        'zs_classify': 0
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "AllInOne"
    seed = 0
    datasets = ["wevid", "howto100m", "yttemporal"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # 128 x 32
    # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    linear_evaluation = False

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 224  # 384/224
    patch_size = 16  # 16/32
    max_image_len = -1
    draw_false_image = 1
    image_only = False
    num_frames = 3  # input video frames

    # Tzxt Setting
    vqav2_label_size = 3129
    msrvttqa_label_size = 1501
    max_text_len = 40  # original: 40, 200: for long sentences/paragraph
    tokenizer = "pretrained/bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    draw_options_text = 0
    # Transformer Setting
    vit = "vit_base_patch16_224"  # "vit_base_patch32_384" / "vit_base_patch16_224"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    shared_embedding_dim = 512  #  add for contrastive learning 512/256
    # model_temporal_frames = 4  #  add for model define， may not consistent with input data

    save_checkpoints_interval = 5  # save each 5 epochs

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads
    backend = 'a100'  # gpu: a100/v100/others

    # Downstream Setting
    get_recall_metric = False
    get_itc_recall_metric = False
    get_ind_recall_metric = False
    retrieval_views = 3  # how many views for retrieval

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16


# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def env_dandelin():
    data_root = "/data2/dsets/dataset"
    log_dir = "/data2/AllInOne/result"
    num_gpus = 8
    num_nodes = 1


@ex.named_config
def task_mlm_itm_webvid():
    exp_name = "mlm_itm"
    datasets = ["webvid"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 1024
    max_epoch = 10
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?


# test additional vcop
@ex.named_config
def task_mlm_itm_vcop_webvid():
    exp_name = "mlm_itm"
    datasets = ["webvid"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "vcop": 1})
    batch_size = 1024
    max_epoch = 10
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?


@ex.named_config
def task_mlm_itm_howto100m():
    exp_name = "mlm_itm"
    datasets = ["howto100m"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 1024
    max_epoch = 10
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?
    val_check_interval = 1.0


@ex.named_config
def task_mlm_itm_yttemporal():
    exp_name = "mlm_itm"
    datasets = ["yttemporal"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 1024
    max_epoch = 10
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?
    val_check_interval = 1.0


@ex.named_config
def task_mlm_itm_webvid_howto():
    exp_name = "mlm_itm_webvid_howto"
    datasets = ["howto100m", "webvid"]  # "howto100m", "webvid",
    loss_names = _loss_names({"itm": 1, "mlm": 1})  # "itc": 1, "itm": 1,
    batch_size = 4096  # max batch size for all nodes 256gpu x 16 = 4096
    max_epoch = 100
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?
    val_check_interval = 1.0  # val for each 0.3 epoch, if int, is sample


@ex.named_config
def task_mlm_ind_itc_webvid_howto():
    exp_name = "mlm_ind_itc_webvid_howto"
    datasets = ["howto100m", "webvid"]  # "howto100m", "webvid",
    loss_names = _loss_names({"ind_itc": 1, "mlm": 1})  # "itc": 1, "itm": 1,
    batch_size = 4096  # max batch size for all nodes 256gpu x 16 = 4096
    max_epoch = 100
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?
    val_check_interval = 1.0  # val for each 0.3 epoch, if int, is sample


@ex.named_config
def task_mlm_itm_webvid_howto_ytt():
    exp_name = "mlm_itm_webvid_howto_ytt"
    datasets = ["howto100m", "webvid", "yttemporal"]  # "howto100m", "webvid",
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # max batch size for all nodes
    max_epoch = 100
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?
    val_check_interval = 1.0  # val for each 0.3 epoch, if int, is sample

@ex.named_config
def task_mlm_itm_image_joint():
    exp_name = "mlm_itm"
    datasets = ["cc3m", "coco", "vg"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 1024  # max batch size for all nodes
    max_epoch = 10
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?
    val_check_interval = 1.0  # val for each 0.3 epoch

# ==== add
@ex.named_config
def task_mlm_itm_vcop_webvid_howto():
    exp_name = "mlm_itm"
    datasets = ["howto100m", "webvid"]  # "howto100m", "webvid",
    loss_names = _loss_names({"itm": 1, "mlm": 1, "vcop": 1})
    batch_size = 2048  # max batch size for all nodes
    max_epoch = 10
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?
    val_check_interval = 1.0  # val for each 0.3 epoch, if int, is sample


@ex.named_config
def task_mlm_itm_ind_itc_webvid_howto():
    exp_name = "mlm_itm"
    datasets = ["howto100m", "webvid"]  # "howto100m", "webvid",
    loss_names = _loss_names({"itm": 1, "mlm": 1, "ind_itc": 1})
    batch_size = 1024  # max batch size for all nodes
    max_epoch = 10
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?
    val_check_interval = 1.0  # val for each 0.3 epoch, if int, is sample

@ex.named_config
def task_mlm_itm_cc_webvid_howto():
    exp_name = "mlm_itm"
    datasets = ["cc3m", "coco", "vg", "howto100m", "webvid"]  # "howto100m", "webvid",
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 1024  # max batch size for all nodes
    max_epoch = 10
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?
    val_check_interval = 1.0  # val for each 0.3 epoch, if int, is sample


@ex.named_config
def task_ind_mlm_itm_cc_webvid_howto():
    exp_name = "mlm_itm"
    datasets = ["cc3m", "coco", "vg", "howto100m", "webvid"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "ind_itc": 1})
    batch_size = 1024  # max batch size for all nodes
    max_epoch = 10
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?
    val_check_interval = 1.0  # val for each 0.3 epoch

@ex.named_config
def task_mlm_itm_ego4d():
    exp_name = "mlm_itm"
    datasets = ["ego4d"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 1024
    max_epoch = 10
    max_image_len = -1    # -1/200 only use 200 image tokens for pretrain?

#  add contrastive learning pipeline
@ex.named_config
def task_mlm_itm_itc_webvid():
    exp_name = "mlm_itm_itc"
    datasets = ["webvid"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
    batch_size = 1024
    max_epoch = 10
    max_image_len = -1 # -1/200


#  add contrastive learning pipeline
@ex.named_config
def task_mlm_itm_ind_itc_webvid():
    exp_name = "mlm_itm_ind_itc"
    datasets = ["webvid"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "ind_itc": 1})
    batch_size = 1024
    max_epoch = 10
    max_image_len = -1 # -1/200


#  add contrastive learning pipeline
@ex.named_config
def task_itm_itc_webvid():
    exp_name = "itm_itc"
    datasets = ["webvid"]
    loss_names = _loss_names({"itm": 1, "itc": 1})
    batch_size = 1024
    max_epoch = 10
    max_image_len = -1 # -1/200


#  add contrastive learning pipeline
@ex.named_config
def task_ind_itc_webvid():
    exp_name = "ind_itc"
    datasets = ["webvid"]
    loss_names = _loss_names({"ind_itc": 1})
    batch_size = 1024
    max_epoch = 10
    max_image_len = -1  # -1/200
    val_check_interval = 0.3
# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
# @ex.named_config
# def task_mlm_itm():
#     exp_name = "mlm_itm"
#     datasets = ["coco", "vg", "sbu", "gcc"]
#     loss_names = _loss_names({"itm": 1, "mlm": 1})
#     batch_size = 4096
#     max_epoch = 10
#     max_image_len = 200


@ex.named_config
def task_mlm_itm():
    exp_name = "mlm_itm"
    datasets = ["cc3m"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 1024
    max_epoch = 10
    max_image_len = -1  # -1 /200


@ex.named_config
def task_mlm_itm_randaug():
    exp_name = "mlm_itm_randaug"
    datasets = ["coco", "vg", "sbu", "gcc"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = -1 # -1 /200


@ex.named_config
def task_mlm_itm_mpp():
    exp_name = "mlm_itm_mpp"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mpp": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = -1 # -1 /200


@ex.named_config
def task_finetune_nlvr2():
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_nlvr2_randaug():
    exp_name = "finetune_nlvr2_randaug"
    datasets = ["nlvr2"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.5
    lr_mult = 10


@ex.named_config
def task_finetune_vqa_randaug():
    exp_name = "finetune_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.5
    lr_mult = 10


# = add by  for msrvtt qa
@ex.named_config
def task_finetune_msrvttqa():
    exp_name = "finetune_msrvtt_qa"
    datasets = ["msrvttqa"]
    loss_names = _loss_names({"openend_vqa": 1})
    batch_size = 512
    msrvttqa_label_size = 1501  # 1501 / 4540
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1  # 0.1
    draw_false_image = 1
    draw_false_text = 1
    learning_rate = 1e-4  # 1e-4 normal
    val_check_interval = 1.0
    lr_mult = 10

# for msvd qa
@ex.named_config
def task_finetune_msvdqa():
    exp_name = "finetune_msvd_qa"
    datasets = ["msvdqa"]
    loss_names = _loss_names({"openend_vqa": 1})  # msvd have same number of answers with msrvtt
    batch_size = 512
    msrvttqa_label_size = 1001  # vqa voculbary length 1000 + 1 background
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4  # 1e-4
    val_check_interval = 1.0
    lr_mult = 10

# = for tgif qa on frameqa
@ex.named_config
def task_finetune_tgifqa():
    exp_name = "finetune_tgif_qa"
    datasets = ["tgif"]
    loss_names = _loss_names({"openend_vqa": 1})
    batch_size = 512
    msrvttqa_label_size = 1541  # vqa voculbary length 1540 + 1 background
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4  # 1e-4
    val_check_interval = 1.0
    lr_mult = 10

# = for tgif qa on action/trans
@ex.named_config
def task_finetune_tgif_action_trans():
    exp_name = "finetune_tgif_action_trans"
    datasets = ["tgifqa"]
    loss_names = _loss_names({"mc_vqa": 1})
    batch_size = 512
    max_epoch = 100
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    draw_options_text = 5  # 5 choices
    learning_rate = 1e-4  # 1e-4
    val_check_interval = 1.0
    lr_mult = 10

# = for msrvtt multiple choice
@ex.named_config
def task_finetune_msrvttchoice():
    exp_name = "finetune_msrvtt_choice"
    datasets = ["msrvtt_choice"]
    loss_names = _loss_names({"multiple_choice": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_text = 5 # 5 choices
    learning_rate = 1e-4
    val_check_interval = 0.5
    lr_mult = 10


# = for lsmdc multiple choice
@ex.named_config
def task_finetune_lsmdcchoice():
    exp_name = "finetune_lsmdc_choice"
    datasets = ["lsmdc_choice"]
    loss_names = _loss_names({"multiple_choice": 1})  # the loss is consistent with msrvtt
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_text = 5  # 5 choices
    learning_rate = 1e-5
    val_check_interval = 0.5
    lr_mult = 10


# = for ego4d multiple choice
@ex.named_config
def task_finetune_ego4dchoice():
    exp_name = "finetune_ego4d_choice"
    datasets = ["ego4d_choice"]
    loss_names = _loss_names({"multiple_choice": 1})   # the loss is consistent with msrvtt
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_text = 4  # 4 choices
    learning_rate = 1e-4
    val_check_interval = 0.5
    lr_mult = 10

# = for vcr q2a
@ex.named_config
def task_finetune_vcrq2a():
    exp_name = "finetune_vcr_q2a"
    datasets = ["vcr"]
    loss_names = _loss_names({"vcr_q2a": 1})
    batch_size = 256
    train_transform_keys = ["pixelbert_randaug"]
    max_epoch = 200
    max_steps = None
    warmup_steps = 0.1
    draw_options_text = 4  # 4 choices
    learning_rate = 1e-4
    val_check_interval = 0.5
    lr_mult = 10

# =  for tvqa
@ex.named_config
def task_finetune_tvqa():
    exp_name = "finetune_tvqa"
    datasets = ["tvqa"]
    loss_names = _loss_names({"mc_vqa": 1})  # tvqa have same number of answers with msrvtt
    batch_size = 512
    max_epoch = 100
    max_text_len = 100  # the text includes options maybe very long
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    draw_options_text = 5  # 5 choices
    learning_rate = 1e-4  # 1e-4
    val_check_interval = 1.0
    lr_mult = 10

# Task4: ===================== Video/Image Text Retrieval =====================
@ex.named_config
def task_finetune_irtr_coco():
    exp_name = "finetune_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_coco_randaug():
    exp_name = "finetune_irtr_coco_randaug"
    datasets = ["coco"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4

@ex.named_config
def task_finetune_ind_itc_irtr_coco_randaug():
    exp_name = "finetune_irtr_coco_randaug"
    datasets = ["coco"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"ind_itc": 1, "itm": 0.5,  "irtr": 1})  # "itm": 0.5,  "irtr": 1
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = False
    get_itc_recall_metric = False
    get_ind_recall_metric = True
    draw_false_text = 15
    learning_rate = 3e-4

#  msvd
@ex.named_config
def task_finetune_irtr_msvd():
    exp_name = "finetune_irtr_msvd"
    datasets = ["msvd"]
    loss_names = _loss_names({"ind_itc": 1, "itm": 0.5, "irtr": 1})  # "itm": 0.5, "irtr": 1,
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = False
    get_itc_recall_metric = False
    get_ind_recall_metric = True
    draw_false_text = 10
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_msrvtt():
    exp_name = "finetune_irtr_msrvtt"
    datasets = ["msrvtt"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4
    # max_image_len = 200


@ex.named_config
def task_finetune_itc_irtr_msrvtt_randaug():
    exp_name = "finetune_itc_irtr_msrvtt_randaug"
    datasets = ["msrvtt"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "itc": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = False
    get_itc_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


# ind itc
@ex.named_config
def task_finetune_only_ind_itc_msrvtt_randaug():
    exp_name = "finetune_itc_irtr_msrvtt_randaug"
    datasets = ["msrvtt"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"ind_itc": 1})
    batch_size = 1024
    max_epoch = 200
    max_steps = None
    warmup_steps = 0.1
    retrieval_views = 3  # use 5 views
    get_recall_metric = False
    get_itc_recall_metric = False
    get_ind_recall_metric = True
    draw_false_text = 15
    learning_rate = 3e-4  # 1/3e-4


# ind itc
@ex.named_config
def task_finetune_ind_itc_irtr_msrvtt_randaug():
    exp_name = "finetune_itc_irtr_msrvtt_randaug"
    datasets = ["msrvtt"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"ind_itc": 1, "irtr": 1, "itm": 0.5})  # , "irtr": 1, "itm": 0.5
    batch_size = 1024
    max_epoch = 200
    max_steps = None
    warmup_steps = 0.1
    retrieval_views = 3  # use 5 views
    get_recall_metric = False
    get_itc_recall_metric = False
    get_ind_recall_metric = True
    draw_false_text = 15
    learning_rate = 3e-4  # 1/3e-4


# ind itc
@ex.named_config
def task_finetune_ind_itc_irtr_activitynet_randaug():
    exp_name = "finetune_itc_irtr_activitynet_randaug"
    datasets = ["activitynet"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"ind_itc": 1})  # , "irtr": 1, "itm": 0.5
    batch_size = 1024
    max_epoch = 200
    max_steps = None
    warmup_steps = 0.1
    retrieval_views = 3  # use 5 views
    get_recall_metric = False
    get_itc_recall_metric = False
    get_ind_recall_metric = True
    draw_false_text = 10
    learning_rate = 3e-4  # 1e-4


# ind itc
@ex.named_config
def task_finetune_ind_itc_irtr_didemo_randaug():
    exp_name = "finetune_itc_irtr_didemo_randaug"
    datasets = ["didemo"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"ind_itc": 1})  # , "irtr": 1, "itm": 0.5
    batch_size = 1024
    max_text_len = 80  # perform paragraph to video matching
    max_epoch = 200
    max_steps = None
    warmup_steps = 0.1
    retrieval_views = 3  # use 5 views
    get_recall_metric = False
    get_itc_recall_metric = False
    get_ind_recall_metric = True
    draw_false_text = 10
    learning_rate = 6e-4  # 1e-4 / 6e-4

# ind itc
@ex.named_config
def task_finetune_ind_itc_irtr_lsmdc_randaug():
    exp_name = "finetune_itc_irtr_lsmdc_randaug"
    datasets = ["lsmdc"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"ind_itc": 1, "irtr": 1, "itm": 0.5})  # , "irtr": 1, "itm": 0.5
    batch_size = 1024
    max_text_len = 80  # perform paragraph to video matching
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    retrieval_views = 3  # use 5 views
    get_recall_metric = False
    get_itc_recall_metric = False
    get_ind_recall_metric = True
    draw_false_text = 10
    learning_rate = 1e-4  # 1e-4 / 6e-4


@ex.named_config
def task_finetune_irtr_msrvtt_randaug():
    exp_name = "finetune_irtr_msrvtt_randaug"
    datasets = ["msrvtt"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 10  # 5/15
    learning_rate = 1e-4



@ex.named_config
def task_finetune_irtr_f30k():
    exp_name = "finetune_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k_randaug():
    exp_name = "finetune_irtr_f30k_randaug"
    datasets = ["f30k"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4

# end: ===================== video/image text retrieval =====================

# Task5: ===================== action recognition =====================
@ex.named_config
def task_finetune_action_recognition_hmdb51():
    exp_name = "finetune_action_recognition_hmdb51"
    datasets = ["hmdb51"]
    loss_names = _loss_names({"openend_vqa": 1})  # have
    msrvttqa_label_size = 52  # 51 + 1
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_action_recognition_k400():
    exp_name = "finetune_action_recognition_k400"
    datasets = ["k400"]
    loss_names = _loss_names({"openend_vqa": 1})  # have
    msrvttqa_label_size = 401  # 400 + 1
    batch_size = 256
    max_epoch = 50
    max_steps = None
    warmup_steps = 0.1
    draw_false_text = 15
    learning_rate = 3e-4
    val_check_interval = 1.0
# end: ===================== action recognition =====================


# Task6: ==================zero-shot action recognition ==============
# end: ====================zero-shot action recognition ==============

# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end


@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000


@ex.named_config
def vit32_base():
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12
