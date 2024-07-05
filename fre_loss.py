import numpy as np
import torch_harmonics as th
import torch
from cv2 import getGaussianKernel


def project_pc_on_unit_sphere(batch, nlat, nlon, knn_k=5, knn_sigma=0.05, return_true_point_indices=False,
                              device='cuda' if torch.cuda.is_available() else 'cpu'):
    ngrid = nlon * nlat
    grid_lon = torch.from_numpy(np.linspace(0, (nlon - 1) * np.pi * 2 / nlon, num=nlon, endpoint=True)).to(device, torch.float32)
    grid_lat = torch.from_numpy(np.linspace(-np.pi / 2, np.pi / 2, num=nlat, endpoint=True)).to(device, torch.float32)
    grid_lon = torch.broadcast_to(grid_lon, (nlat, nlon))
    grid_lat = torch.broadcast_to(grid_lat.view(nlat, 1), (nlat, nlon))
    grid_lon = grid_lon.reshape(1, 1, ngrid)
    grid_lat = grid_lat.reshape(1, 1, ngrid)

    origin = batch.mean(dim=1, keepdim=True)  # the center of the unit sphere
    normalized_batch = batch - origin  # for looking from the perspective of the origin
    B, npc, _ = batch.shape

    pc_x, pc_y, pc_z = normalized_batch[..., 0], normalized_batch[..., 1], normalized_batch[..., 2]
    pc_r = torch.sqrt(pc_x * pc_x + pc_y * pc_y + pc_z * pc_z)
    pc_lat = torch.arcsin(pc_z / pc_r)
    pc_lon = torch.atan2(pc_y, pc_x) % (2 * torch.pi)
    pc_lat = pc_lat.reshape(B, npc, 1)
    pc_lon = pc_lon.reshape(B, npc, 1)

    pre_dist = torch.cos(grid_lat) * torch.cos(pc_lat) * torch.cos(grid_lon - pc_lon) + torch.sin(grid_lat) * torch.sin(pc_lat)  # [B, N, ngrid]
    squared_dist = 2 - 2 * pre_dist  # [B, N, ngrid]

    _, inds = squared_dist.topk(k=knn_k, dim=1, largest=False)  # inds: [B, knn_k, ngrid]
    gaussian_coeffs = (-squared_dist.gather(dim=1, index=inds) / (2 * knn_sigma ** 2)).exp()

    zero_threshold = 1e-10
    zero_inds = (gaussian_coeffs < zero_threshold)
    gaussian_coeffs = gaussian_coeffs - zero_inds * gaussian_coeffs
    zero_rows_inds = zero_inds.all(dim=1, keepdim=True).expand(-1, knn_k, -1)
    gaussian_coeffs = gaussian_coeffs + zero_rows_inds * (1 / knn_k)
    gaussian_coeffs = gaussian_coeffs / gaussian_coeffs.sum(dim=1, keepdim=True)
    grids = (gaussian_coeffs * pc_r.unsqueeze(-1).expand(-1, -1, ngrid).gather(dim=1, index=inds)).sum(dim=1)
    grids = grids.reshape(B, nlat, nlon)
    if return_true_point_indices:
        true_point_indices = squared_dist.argmin(dim=2)  # argmin on a different axis [B, N]
        return grids, true_point_indices
    return grids


def extract_high_freq(batch, lmax=50, sigma=50, device="cuda" if torch.cuda.is_available() else "cpu"):
    nlat = 2 * lmax + 2
    nlon = nlat * 2
    nlat += 1
    nlon += 1

    data = project_pc_on_unit_sphere(batch, nlat, nlon, device=device)
    sht = th.RealSHT(nlat, nlon, lmax=lmax + 1, grid="equiangular").to(device)
    coeffs = sht(data)[:, :, :lmax + 1]
    coeffs = torch.stack([coeffs.real, coeffs.imag], dim=1)
    weights = torch.from_numpy(getGaussianKernel(coeffs.shape[2] * 2 - 1, sigma)[coeffs.shape[2] - 1::-1]
                               .reshape(-1).copy()).to(device, torch.float32)

    coeffs *= weights / weights[-1]
    all_coeffs = []
    for i in range(lmax + 1):
        all_coeffs.append(coeffs[:, 0, i, :i + 1])
        all_coeffs.append(coeffs[:, 1, i, :i])
    return torch.cat(all_coeffs, dim=1)


def fre_loss(batch1, batch2, lmax=50, sigma=50):
    assert batch1.shape == batch2.shape
    coeffs1 = extract_high_freq(batch1, lmax=lmax, sigma=sigma)
    coeffs2 = extract_high_freq(batch2, lmax=lmax, sigma=sigma)
    return torch.nn.functional.mse_loss(coeffs1, coeffs2)
