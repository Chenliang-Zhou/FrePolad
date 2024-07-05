import os
import time

import torch
import torch.nn as nn
from models import get_latent_ddpm, get_vae

from args import parse_args
from dataset import ShapeNetPointCloud15K
from utils import prepare_training, requires_grad


def main_worker(local_rank, args):
    prepare_training(local_rank, args)
    loss_file = os.path.join(args.out_dir, "loss.txt")

    print("Loading model...")
    if args.model == "vae":
        model = get_vae(args)
    else:  # ldm
        vae = get_vae(args).eval()
        vae.load_state_dict(torch.load(args.vae_path, map_location=args.device)["model_state_dict"], strict=True)
        requires_grad(vae, False)
        model = get_latent_ddpm(args)

    if args.num_gpus > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    starting_epoch = 1
    losses = []
    epochs = []

    # resume checkpoints
    if args.inference:
        assert args.resume_path is not None, "cannot find network weights to resume from."
    if args.resume_path is not None:
        print(f"Resuming from pretrained path {args.resume_path}...")
        checkpoint = torch.load(args.resume_path, map_location=args.device)
        if args.num_gpus > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        losses = [t.cpu() for t in checkpoint.get('losses', [])]
        epochs = checkpoint.get('epochs', [])
        loss = checkpoint['loss']
        print("==> Resumed from:", args.resume_path)
        print("==> Starting epoch:", starting_epoch)
        print("==> Previous loss: {:.5f}".format(loss))
    print("Model loaded.")

    print("Loading datasets...")
    dataset = ShapeNetPointCloud15K(args, n=args.num_samples_for_training)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size_per_core, shuffle=True)
    num_batches = len(dataloader)
    print("Dataset loaded.")

    print("Start training....")
    training_start_time = time.time()
    if args.inference:
        args.num_epochs = starting_epoch
    for epoch in range(starting_epoch, args.num_epochs + 1):
        start_time = time.time()
        epoch_loss = torch.tensor(0.)
        if not args.inference:
            model.train()
            for batch in dataloader:
                optimizer.zero_grad()
                if args.model == "vae":
                    loss = model(batch["true_pc"], step=epoch)
                else:  # ldm
                    latent = vae.encode(batch)
                    loss = model(latent, None)

                epoch_loss += loss.detach().cpu()
                loss.backward()
                optimizer.step()

            epoch_loss /= num_batches
            lr_scheduler.step(epoch_loss)
            losses.append(epoch_loss)
            epochs.append(epoch)

        # logs
        if args.global_rank == 0:
            # output loss
            if not args.inference and (epoch == args.num_epochs or (args.log_freq and epoch % args.log_freq == 0)):
                log_message = "Epoch {}: loss: {:.5f}, time for this epoch: {:.5f}s, time so far: {:.5f}h" \
                    .format(epoch, epoch_loss.item(), time.time() - start_time, (time.time() - training_start_time) / 3600)

                with open(loss_file, 'a') as file:
                    file.write(log_message + "\n")
                print(log_message)

            # saving weights
            if not args.inference and (epoch == args.num_epochs or
                                       (args.ckpt_freq and epoch % args.ckpt_freq == 0)):
                print("==> Saving checkpoint...")
                save_path = os.path.join(args.out_dir, f"{args.model}-ckpt-epoch-{epoch}.pt")
                torch.save({'epoch': epoch,
                            'model_state_dict': model.module.state_dict() if args.num_gpus > 1 else model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                            'epochs': epochs,
                            'losses': losses},
                           save_path)
                print(f"====> Saved checkpoint to {save_path}")

    print("Training ended.")


if __name__ == '__main__':
    args = parse_args()
    if args.num_gpus <= 1:
        main_worker(local_rank=0, args=args)
    else:
        torch.multiprocessing.set_start_method('spawn', force=True)
        torch.multiprocessing.spawn(fn=main_worker, args=(args,), nprocs=args.num_gpus_per_node)
    print(f"Main program ended on node {args.node_id}.")
