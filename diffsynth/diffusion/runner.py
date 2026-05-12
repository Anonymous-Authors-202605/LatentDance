import os, torch, sys
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger

import re

def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    exp_name = os.environ.get("EXP_NAME", "unknown")
    exp_save_tag = os.environ.get("EXP_NAME_SAVE_TAG", "")
    # exp_info = f"[{exp_save_tag}/{exp_name}]" if exp_save_tag else f"[{exp_name}]"
    exp_info = f"[{exp_name}]"

    if getattr(args, "resume_checkpoint_path", None) is not None:
        # parse epoch from checkpoint path
        match = re.search(r'epoch-(\d+)\.safetensors', args.resume_checkpoint_path)
        if match:
            resume_epoch = int(match.group(1))
            print(f"{exp_info} Resuming from epoch {resume_epoch}", flush=True) if accelerator.is_main_process else None
            start_epoch = resume_epoch + 1 
    else:
        start_epoch = 0

    if accelerator.is_main_process:
        print("Preparing training task...", flush=True)

    train_batch_size = getattr(args, "train_batch_size", 1) if args is not None else 1
    batch_sampler = getattr(args, "_batch_sampler", None) if args is not None else None
    batch_mode = os.environ.get("BATCH_MODE", "true_batch").lower()

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    if train_batch_size > 1:
        if batch_sampler is not None and batch_mode != "micro_batch":
            # Use orientation-grouped batch sampler to ensure same-shape batches (true_batch only)
            dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=lambda x: x, num_workers=num_workers)
        else:
            # micro_batch mode or no batch_sampler: plain shuffled DataLoader
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True, collate_fn=lambda x: x, num_workers=num_workers)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    # When using batch_sampler, DataLoader.batch_size is None which causes DeepSpeed
    # to fail. Explicitly set train_micro_batch_size_per_gpu in the DeepSpeed config.
    if batch_sampler is not None and accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = train_batch_size

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    if accelerator.is_main_process:
        batch_mode = os.environ.get("BATCH_MODE", "true_batch").lower()
        print(f"{exp_info} Training started. Epochs: {num_epochs}, Steps per epoch: {len(dataloader)}, Batch size per GPU: {train_batch_size}, Batch mode: {batch_mode}", flush=True)

    load_from_cache = False
    if hasattr(dataset, "load_from_cache"):
        load_from_cache = dataset.load_from_cache
    elif hasattr(dataset, "datasets") and len(dataset.datasets) > 0:
        load_from_cache = getattr(dataset.datasets[0], "load_from_cache", False)

    for epoch_id in range(start_epoch, num_epochs):
        for data in tqdm(dataloader, desc=f"{exp_info} Epoch {epoch_id+1}/{num_epochs}", disable=not accelerator.is_main_process, file=sys.stdout):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    if accelerator.is_main_process:
        print(f"Data processing started. Total steps: {len(dataloader)}", flush=True)

    for data_id, data in enumerate(tqdm(dataloader, disable=not accelerator.is_main_process, file=sys.stdout)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
