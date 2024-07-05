import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNetPointCloud15K(Dataset):
    def __init__(self, args,
                 tr_sample_size=2048,
                 te_sample_size=2048,
                 scale=1.,
                 normalize_per_shape=False,
                 normalize_shape_box=False,
                 random_subsample=True,
                 sample_with_replacement=1,
                 normalize_std_per_axis=False,
                 normalize_global=True,
                 recenter_per_shape=False,
                 all_points_mean=None,
                 all_points_std=None,
                 input_dim=3,
                 n=None,
                 randomize=True,
                 ):
        self.randomize = randomize

        self.normalize_shape_box = normalize_shape_box
        self.path = args.dataset
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.device = args.device

        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.scale = scale
        self.random_subsample = random_subsample
        self.sample_with_replacement = sample_with_replacement
        self.input_dim = input_dim

        self.all_points = []
        self.all_renderings = []
        self.all_rendering_latents = []

        if n is None:
            all_mids = sorted([x[:-4] for x in os.listdir(args.dataset) if x.endswith('.npy')])
            self.n = len(all_mids)
        else:
            self.n = n
            all_mids = sorted([x[:-4] for x in os.listdir(args.dataset) if x.endswith('.npy')])[:n]

        for mid in all_mids:
            obj_fname = os.path.join(args.dataset, mid + ".npy")
            point_cloud = np.load(obj_fname)  # (15k, 3)
            self.all_points.append(point_cloud[np.newaxis, ...])

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random().shuffle(self.shuffle_idx)
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.recenter_per_shape = recenter_per_shape
        if self.normalize_shape_box:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = (  # B,1,3
                                           (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) +
                                           (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)) / 2
            self.all_points_std = np.amax(  # B,1,1
                ((np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) -
                 (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)),
                axis=-1).reshape(B, 1, 1) / 2
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(
                B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(
                    B, -1).std(axis=1).reshape(B, 1, 1)
        elif all_points_mean is not None and all_points_std is not None and not self.recenter_per_shape:
            # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.recenter_per_shape:  # per shape center
            B, N = self.all_points.shape[:2]
            self.all_points_mean = (
                                           (np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) +
                                           (np.amin(self.all_points, axis=1)).reshape(B, 1,
                                                                                      input_dim)) / 2
            self.all_points_std = np.amax(
                ((np.amax(self.all_points, axis=1)).reshape(B, 1, input_dim) -
                 (np.amin(self.all_points, axis=1)).reshape(B, 1, input_dim)),
                axis=-1).reshape(B, 1, 1) / 2
        # else:  # normalize across the dataset
        elif normalize_global:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    -1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)
        else:
            raise NotImplementedError('No Normalization')
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std

        # Default display axis order
        self.display_axis_order = [0, 1, 2]
        print(f"==> Loaded {len(self.all_points)} point clouds.")

    def get_pc_stats(self, idx):
        if self.recenter_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        if self.normalize_per_shape or self.normalize_shape_box:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), \
            self.all_points_std.reshape(1, -1)

    def get_all_pc_mean(self):
        return self.all_points_mean

    def get_all_pc_std(self):
        return self.all_points_std

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + \
                          self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / \
                          self.all_points_std
        self.train_points = self.all_points[:, :min(
            10000, self.all_points.shape[1])]

    def __len__(self):
        return len(self.all_points)

    def __getitem__(self, idx):
        if self.random_subsample and self.sample_with_replacement:
            tr_idxs = np.random.choice(15000, self.tr_sample_size)
        elif self.random_subsample and not self.sample_with_replacement:
            tr_idxs = np.random.permutation(
                np.arange(15000))[:self.tr_sample_size]
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        return torch.from_numpy(self.all_points[idx, tr_idxs, :]).float().to(self.device)