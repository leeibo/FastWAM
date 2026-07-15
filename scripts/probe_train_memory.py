import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from fastwam.runtime import build_datasets, create_fastwam, run_training  # noqa: F401
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import setup_logging

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), f)

    from hydra.utils import instantiate
    from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision, _resolve_train_device

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )

    trainer._set_dit_only_train_mode()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    data_iter = iter(trainer.train_loader)
    sample = next(data_iter)
    actual_batch_size = int(sample["video"].shape[0])
    with trainer.accelerator.accumulate(trainer.model):
        train_model = trainer.model if hasattr(trainer.model, "training_loss") else trainer.accelerator.unwrap_model(trainer.model)
        with trainer.accelerator.autocast():
            loss, loss_dict = train_model.training_loss(sample)
        trainer.accelerator.backward(loss)
        if trainer.accelerator.sync_gradients:
            grad_norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)
            trainer.optimizer.step()
            if not trainer.accelerator.optimizer_step_was_skipped:
                trainer.scheduler.step()
            trainer.optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = torch.tensor(0.0, device=loss.device)

    trainer.accelerator.wait_for_everyone()
    device = trainer.accelerator.device
    if torch.cuda.is_available():
        local = torch.tensor(
            [
                float(torch.cuda.max_memory_allocated(device) / (1024**3)),
                float(torch.cuda.max_memory_reserved(device) / (1024**3)),
                float(torch.cuda.memory_allocated(device) / (1024**3)),
                float(torch.cuda.memory_reserved(device) / (1024**3)),
                float(loss.detach().float().item()),
                float(grad_norm.detach().float().item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                float(actual_batch_size),
            ],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
    else:
        local = torch.zeros((1, 7), device=device, dtype=torch.float32)

    gathered = trainer.accelerator.gather(local)
    if trainer.accelerator.is_main_process:
        rows = gathered.detach().cpu().tolist()
        print("[probe] batch_size_per_gpu=%s world_size=%d" % (cfg.batch_size, trainer.accelerator.num_processes), flush=True)
        for rank, row in enumerate(rows):
            print(
                "[probe] rank=%d actual_batch=%d max_alloc_gib=%.2f max_reserved_gib=%.2f final_alloc_gib=%.2f final_reserved_gib=%.2f loss=%.4f grad_norm=%.4f"
                % (rank, int(row[6]), row[0], row[1], row[2], row[3], row[4], row[5]),
                flush=True,
            )

    trainer.accelerator.wait_for_everyone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
