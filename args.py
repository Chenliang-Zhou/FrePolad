import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="FrePolad")
    # dataset
    parser.add_argument('--dataset', type=str, help="path to the dataset")

    # training options
    parser.add_argument("-i", "--inference", help="whether running inference", action="store_true")
    parser.add_argument("--num-epochs", help="number of epochs for training", type=int, default=200)
    parser.add_argument('--batch-size', help='total batch size', type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-5)
    parser.add_argument("--out-dir", help="output directory", type=str)
    parser.add_argument('--resume-path', help='resume training from given network path', type=str)

    # logging
    parser.add_argument('--num-logs', help='number of times to output loss during training (will always output loss at the end of training)',
                        type=int, default=9999)
    parser.add_argument('--num-ckpt', help='number of checkpoints during training (will always checkpoint at the end of training)', type=int,
                        default=0)

    # distributed training
    parser.add_argument('--num-nodes', help='number of nodes to use', type=int, default=int(os.environ.get("SLURM_JOB_NUM_NODES", 1)))
    parser.add_argument('--node-id', help='the node ID for this process', type=int, default=int(os.environ.get("SLURM_NODEID", 0)))
    parser.add_argument('--num-gpus-per-node', help='number of GPUs to use per node', type=int, default=torch.cuda.device_count())
    parser.add_argument('--device', help='Device to use', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # network
    parser.add_argument('--model', help="The model to be trained", choices=["vae", "ldm"], default="vae")
    parser.add_argument('--data-class', help="The class of the point clouds", choices=["plane", "chair", "car"], default="plane")
    parser.add_argument('--latent-dim', help="The dimensionality of latent space.", type=int, default=1024)
    parser.add_argument('--num-points-input', help="Number of points in the point clouds in training set", type=int, default=2048)
    parser.add_argument('--ldm-backbone', help="The backbone of ldm", choices=["unet1", "unet1x", "unet1024"], default="unet1x")
    parser.add_argument('--vae-cnf-dims', help="Dimensions of VAE CNF hidden layers", type=str, default="512-512-512")
    parser.add_argument('--vae-max-kl-weight',
                        help="Maximum KL weight during VAE training (KL increases linearly from --vae-min-kl-weight to --vae-max-kl-weight"
                             "during the first half of training and then stays at --vae-max-kl-weight in the second half."
                             "Can also be used together with --vae-cyclic-kl-annealing", type=float, default=1.0)
    parser.add_argument('--vae-min-kl-weight',
                        help="Minimum KL weight during VAE training (KL increases linearly from --vae-min-kl-weight to --vae-max-kl-weight"
                             "during the first half of training and then stays at --vae-max-kl-weight in the second half."
                             "Can also be used together with --vae-cyclic-kl-annealing", type=float, default=1e-7)
    parser.add_argument('--vae-cyclic-kl-annealing',
                        help="Whether to use cyclic KL annealing during VAE training (4 cycles; ratio for increasing weight is 0.5)",
                        action="store_true")
    parser.add_argument('--vae-high-freq-recon-coeff', help="Coefficient for high frequency reconstruction loss for VAE", default=0, type=float)
    parser.add_argument('--vae-high-freq-recon-lmax', help="The max degree for high frequency reconstruction loss for VAE", default=50, type=int)
    parser.add_argument('--vae-path', help="The path to autoencoder (or vae) network (only useful when --model==ldm_ac or ldm_vae)", type=str)

    args = parser.parse_args()

    if args.inference:
        args.num_nodes = 1
        args.num_gpus_per_node = min(1, args.num_gpus_per_node)
    args.num_gpus = args.num_gpus_per_node * args.num_nodes
    if args.num_gpus > 0:
        assert args.batch_size % args.num_gpus == 0, "batch-size must be a multiple of total number of GPUs"
        args.batch_size_per_core = args.batch_size // args.num_gpus
    else:
        args.batch_size_per_core = args.batch_size
    if args.num_logs == 0:
        args.log_freq = 0
    else:
        args.log_freq = max(args.num_epochs // args.num_logs, 1)
    if args.num_ckpt == 0:
        args.ckpt_freq = 0
    else:
        args.ckpt_freq = max(args.num_epochs // args.num_ckpt, 1)

    return args
