import builtins
import datetime
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def no_print(*args, **kwargs):
    return


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_dict_to_file(d, filename):
    with open(filename, 'wt') as f:
        json.dump(d, f, indent=2)


def prepare_training(local_rank, args):
    global_rank = args.node_id * args.num_gpus_per_node + local_rank
    if global_rank != 0:
        builtins.print = no_print

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"In {__file__}. Experiment options:")
    print(json.dumps(args.__dict__, indent=2))

    set_random_seed(global_rank)
    original_print = print

    if global_rank == 0:
        save_dict_to_file(args.__dict__, os.path.join(args.out_dir, 'training_options.json'))

    print("Initializing torch.distributed...")
    if args.num_gpus > 1:
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', "12340")
        print(f"MASTER_ADDR={os.environ['MASTER_ADDR']}")
        print(f"MASTER_PORT={os.environ['MASTER_PORT']}")
        dist.init_process_group(backend='nccl', rank=global_rank, world_size=args.num_gpus, timeout=datetime.timedelta(minutes=5))

    if args.num_gpus >= 1:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        builtins.print = original_print
        print(f"GPU set up OK: global rank: {global_rank}, local rank: {local_rank} on node {args.node_id}")
        if global_rank != 0:
            builtins.print = no_print
    else:
        device = "cpu"
        print("No GPU detected. Using CPU.")
    args.device = device
    args.global_rank = global_rank


def requires_grad(model, requires_grad=True):
    for p in model.parameters():
        p.requires_grad = requires_grad


def ldm_sampling(model, batch_size, cond=None, verbose=False, seeds=None, latent_dim=1024):
    latent_shape = (latent_dim,)
    if seeds is None:
        seeds = range(batch_size)
    x_T = torch.stack([torch.randn(latent_shape, generator=torch.Generator().manual_seed(seed)) for seed in seeds], dim=0).to(model.device)
    return model.p_sample_loop(cond, shape=(batch_size, *latent_shape), x_T=x_T, verbose=verbose)
