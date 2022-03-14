import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from AllInOne.modules.dist_utils import all_gather
from AllInOne.modules.objectives import compute_irtr_recall, compute_decouple_irtr_recall, compute_ind_irtr_recall
from AllInOne.gadgets.my_metrics import Accuracy, VQAScore, Scalar


def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            # vcr
            elif k == "vcr_q2a":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_vcr_qar_loss", Scalar())
                setattr(pl_module, f"{split}_vcr_qar_accuracy", Accuracy())
            elif k == "mc_vqa":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
            elif k == "openend_vqa":
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_vqa_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
            elif k == "vcop":
                setattr(pl_module, f"{split}_vcop_score", VQAScore())
                setattr(pl_module, f"{split}_vcop_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{k}_accuracy", Accuracy())
            elif k == "multiple_choice":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "nlvr2":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "irtr":
                setattr(pl_module, f"{split}_irtr_loss", Scalar())
            elif k == "mppd" or k == "mpfr":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_wpa_loss", Scalar())
            # add for image text contrastive learning
            elif k == "itc":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0
    the_metric_qar = 0
    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module)
        if torch.distributed.get_rank() == 0:
            print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        the_metric += ir_r1.item() + tr_r1.item()

    # add for decouple irtr
    if pl_module.hparams.config["get_itc_recall_metric"] and not pl_module.training:
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_decouple_irtr_recall(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        # the_metric += ir_r1.item() + tr_r1.item()
        the_metric += ir_r1 + tr_r1
    # == end

    # add for ind irtr
    if pl_module.hparams.config["get_ind_recall_metric"] and not pl_module.training:
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_ind_irtr_recall(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        # the_metric += ir_r1.item() + tr_r1.item()
        the_metric += ir_r1 + tr_r1
    # == end

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        value = 0
        qar_value = 0
        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "vcr_q2a":
            #q2a
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            # qar
            pl_module.log(
                f"vcr_qar/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_vcr_qar_loss").compute(),
            )
            getattr(pl_module, f"{phase}_vcr_qar_loss").reset()
            qar_value = getattr(pl_module, f"{phase}_vcr_qar_accuracy").compute()
            pl_module.log(f"vcr_qar/{phase}/accuracy_epoch", qar_value)
            getattr(pl_module, f"{phase}_vcr_qar_accuracy").reset()
        # mc_vqa
        elif loss_name == "mc_vqa":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
        elif loss_name == "openend_vqa":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_vqa_loss").compute(),
            )
            getattr(pl_module, f"{phase}_vqa_loss").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
        # multiple_choice
        elif loss_name == "multiple_choice":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
        # vcop
        elif loss_name == "vcop":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )

            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
        elif loss_name == "nlvr2":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()

                value = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()
        elif loss_name == "irtr":
            pl_module.log(
                f"{loss_name}/{phase}/irtr_loss_epoch",
                getattr(pl_module, f"{phase}_irtr_loss").compute(),
            )
            getattr(pl_module, f"{phase}_irtr_loss").reset()
        elif loss_name == "mppd" or loss_name == "mpfr":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            pl_module.log(
                f"{loss_name}/{phase}/wpa_loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").reset()
        # add for itc
        elif loss_name == "itc":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        # add for ind_itc
        elif loss_name == "ind_itc":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            # value = f"{loss_name}/{phase}/loss_epoch",
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        if loss_name == "vcr_q2a":
            # print(value, qar_value)
            the_metric += qar_value/2 + value/2
        else:
            the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    names = [n for n, p in pl_module.named_parameters()]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )
    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )


def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):
        undo_dp = True
    elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
        redo_dp = True

    if undo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict