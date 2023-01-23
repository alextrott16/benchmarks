# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import wandb
from composer.utils import dist, reproducibility
from composer import algorithms
from composer import Trainer
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler,
                                      LinearWithWarmupScheduler)
from transformers import Adafactor
from omegaconf import OmegaConf as om

from src.data_c4 import build_c4_dataloader
from src.hf_t5 import create_hf_t5
from src.hf_prefix_lm import create_hf_prefix_lm
from src.inverse_sqrt_scheduler import InverseSquareRootScheduler
from src.mod_print_callback import MixtureOfDenoisersPrinterCallback
from src.super_glue.data import build_super_glue_task_dataloader
from src.momo1.model import ComposerMosaicModel


def build_logger(name, kwargs):
    if name == 'wandb':
        return WandBLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')

def build_callback(name, kwargs):
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor(window_size=kwargs.get('window_size', 1))
    elif name == 'mod_printer':
        return MixtureOfDenoisersPrinterCallback(**kwargs)
    else:
        raise ValueError(f'Not sure how to build callback: {name}')

def build_algorithm(name, kwargs):
    if name == 'fused_layernorm':
        return algorithms.FusedLayerNorm(**kwargs)
    elif name == 'gradient_clipping':
        return algorithms.GradientClipping(**kwargs)
    else:
        raise ValueError(f'Not sure how to build algorithm: {name}')

def build_optimizer(cfg, model):
    if cfg.name == 'decoupled_adamw':
        return DecoupledAdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay
        )
    elif cfg.name == 'adafactor':
        return Adafactor(
            params=model.parameters(),
            lr=cfg.get('lr', 1.0), # Recommend using InverseSquareRootScheduler with default settings when using these defaults
            weight_decay=cfg.get('weight_decay', 0.0),
            beta1=cfg.get('beta1', None),
            scale_parameter=cfg.get('scale_parameter', True),
            relative_step=cfg.get('relative_step', False),
            warmup_init=cfg.get('warmup_init', False)
        )
    else:
        raise ValueError(f'Not sure how to build optimizer: {cfg.name}')

def build_scheduler(cfg):
    if cfg.name == 'constant_with_warmup':
        return ConstantWithWarmupScheduler(
            t_warmup=cfg.t_warmup)
    elif cfg.name == 'linear_decay_with_warmup':
        return LinearWithWarmupScheduler(
            t_warmup=cfg.t_warmup,
            alpha_f=cfg.alpha_f
        )
    elif cfg.name == 'cosine_with_warmup':
        return CosineAnnealingWithWarmupScheduler(
            t_warmup=cfg.t_warmup,
            alpha_f=cfg.alpha_f)
    elif cfg.name == 'inverse_square_root':
        return InverseSquareRootScheduler(
            alpha_max=cfg.alpha_max,
            scale=cfg.get('scale', 1.0),
        )
    else:
        raise ValueError(f'Not sure how to build scheduler: {cfg.name}')

def build_model(cfg):
    if cfg.name == 'hf_t5':
        return create_hf_t5(
            pretrained_model_name=cfg.pretrained_model_name,
            use_pretrained=cfg.get('use_pretrained', None),
            model_config=cfg.get('model_config', None),
            tokenizer_name=cfg.get('tokenizer_name', None),
            z_loss=cfg.get('z_loss', 0.0),
            task_finetuning=cfg.get('task_finetuning', False),
        )
    elif cfg.name == 'hf_prefix_lm':
        return create_hf_prefix_lm(
            pretrained_model_name=cfg.pretrained_model_name,
            tokenizer_name=cfg.tokenizer_name,
            use_pretrained=cfg.get('use_pretrained', None),
            model_config=cfg.get('model_config', None),
            # z_loss=cfg.get('z_loss', 0.0),
            # task_finetuning=cfg.get('task_finetuning', False),
        )
    elif cfg.name == 'mosaic_model':
        return ComposerMosaicModel(cfg)
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')

def build_dataloader(cfg, device_batch_size, mode):
    if cfg.name == 'c4':
        return build_c4_dataloader(cfg, device_batch_size)
    elif cfg.name == 'super_glue':
        return build_super_glue_task_dataloader(cfg, device_batch_size, mode)
    else:
        raise ValueError(f'Not sure how to build dataloader with name={cfg.name}')


def main(cfg):
    print("Training using config: ")
    print(om.to_yaml(cfg))
    reproducibility.seed_all(cfg.seed)

    # Read FSDP Config as a dict
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None

    # Build Model
    print('Initializing model...')
    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params=:.4e}')

    # Get batch size info
    if cfg.global_train_batch_size % dist.get_world_size() != 0:
        raise ValueError(f'Global batch size {cfg.global_train_batch_size} is not divisible by {dist.get_world_size()} '
                         'as a result, the batch size would be truncated, please adjust `global_train_batch_size` '
                         f'to be divisible by world size, {dist.get_world_size()}.')
    device_train_batch_size = cfg.global_train_batch_size // dist.get_world_size()
    device_eval_batch_size = cfg.get('global_eval_batch_size', cfg.global_train_batch_size) // dist.get_world_size()

    # Dataloaders
    print("Building train loader...")
    train_loader = build_dataloader(cfg.train_loader, device_train_batch_size, mode='train')
    print("Building eval loader...")
    eval_loader = build_dataloader(cfg.eval_loader, device_eval_batch_size, mode='eval')

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)
    
    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [build_logger(name, logger_cfg) for name, logger_cfg in cfg.get('loggers', {}).items()]

    # Callbacks
    callbacks = [build_callback(name, callback_cfg) for name, callback_cfg in cfg.get('callbacks', {}).items()]

    # Algorithms
    algorithms = [build_algorithm(name, algorithm_cfg) for name, algorithm_cfg in cfg.get('algorithms', {}).items()]

    if 'run_name' in cfg:
        run_name = cfg['run_name']
    else:
        run_name = os.environ['COMPOSER_RUN_NAME']

    # Build the Trainer
    trainer = Trainer(
        run_name=run_name,
        seed=cfg.seed,
        model=model,
        algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        train_subset_num_batches=cfg.get('train_subset_num_batches', -1),
        eval_subset_num_batches=cfg.get('eval_subset_num_batches', 1000),
        fsdp_config=fsdp_config,  # type: ignore
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        progress_bar=cfg.progress_bar,
        log_to_console=cfg.log_to_console,
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        device=cfg.get('device', None),
        grad_clip_norm=cfg.get('grad_clip_norm', -1.0),
        grad_accum=cfg.get('grad_accum', 'auto'),
        save_folder=cfg.get('save_folder', None),
        save_interval=cfg.get('save_interval', '1000ba'),
        save_num_checkpoints_to_keep=cfg.get('save_num_checkpoints_to_keep', -1),
        load_path=cfg.get('load_path', None),
        load_weights_only=cfg.get('load_weights_only', False),
    )

    print("Logging config...")
    config_dict = om.to_container(cfg, resolve=True)
    config_dict.update({
        'n_gpus': dist.get_world_size(),
        'n_params': n_params,
        'device_train_batch_size': device_train_batch_size,
        'device_eval_batch_size': device_eval_batch_size,
    })
    if wandb.run is not None:
        wandb.config.update(config_dict)

    print("Starting training...")
    trainer.fit()


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
