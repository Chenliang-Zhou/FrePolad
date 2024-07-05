def get_vae(args):
    from PointFlow.models.networks import PointFlow
    from PointFlow.args import get_args
    from LION.models.shapelatent_modules import PointNetPlusEncoder

    vae_args = get_args()
    vae_args.device = args.device
    vae_args.zdim = args.latent_dim
    vae_args.encoder = PointNetPlusEncoder(args.latent_dim, 3).to(args.device)
    vae_args.num_epochs = args.num_epochs
    vae_args.max_kl_weight = args.vae_max_kl_weight
    vae_args.min_kl_weight = args.vae_min_kl_weight
    vae_args.cyclic_kl_annealing = args.vae_cyclic_kl_annealing
    vae_args.high_freq_recon_coeff = args.vae_high_freq_recon_coeff
    vae_args.high_freq_recon_lmax = args.vae_high_freq_recon_lmax
    return PointFlow(vae_args).to(args.device)


def get_latent_ddpm(args):
    from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion

    if args.ldm_backbone == "unet1":
        diff_model_config = {"target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                             "params": {"dims": 1, "in_channels": 1, "model_channels": 256, "up_down_sampling": True,
                                        "attention_resolutions": (2, 4, 8), "channel_mult": (1, 2, 2, 4), "num_res_blocks": 2}}
    elif args.ldm_backbone == "unet1x":
        diff_model_config = {"target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                             "params": {"dims": 1, "in_channels": 1, "model_channels": 320, "up_down_sampling": True,
                                        "attention_resolutions": (2, 4, 8), "channel_mult": (1, 2, 4, 4), "num_res_blocks": 3}}
    elif args.ldm_backbone == "unet1024":
        diff_model_config = {"target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                             "params": {"dims": 1, "in_channels": 1024, "model_channels": 1024,
                                        "up_down_sampling": False}}
    else:
        raise NotImplementedError
    return LatentDiffusion(diff_model_config=diff_model_config, conditioning_key=None).to(args.device)
