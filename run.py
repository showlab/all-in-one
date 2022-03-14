import os
import copy
import pytorch_lightning as pl
import torch
from AllInOne.config import ex
from AllInOne.modules import AllinoneTransformerSS
from AllInOne.datamodules.multitask_datamodule import MTDataModule
import datetime

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)
    model = AllinoneTransformerSS(_config)

    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        # every_n_epochs=_config["save_checkpoints_interval"],
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    now = datetime.datetime.now()
    instance_name = f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}{now.year}_{now.month}_{now.day}'
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=instance_name,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )
    # print all config at the begin
    print('='*70+'Config: '+'='*70)
    print(instance_name)
    print(_config)
    print('='*150)

    # notice _config["batch_size"] should be max length for all machines, eg. at least 1024
    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        # prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        # show_progress_bar=False,
        # progress_bar_refresh_rate=0
    )

    print("accumulate grad batches is: ", trainer.accumulate_grad_batches)

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
