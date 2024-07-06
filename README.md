# FrePolad: Frequency-Rectified Point Latent Diffusion for Point Cloud Generation

<p align="center"><a href="https://chenliang-zhou.github.io">Chenliang Zhou</a>, <a href="https://www.cl.cam.ac.uk/~fz261/">Fangcheng Zhong</a>, <a href="https://paramhanji.github.io">Param Hanji</a>, <a href="https://zhilinguo.github.io">Zhilin Guo</a></p>

<p align="center"><a href="https://kyle-fogarty.github.io">Kyle Fogarty</a>, <a href="https://asztr.github.io">Alejandro Sztrajman</a>, <a href="https://www.cst.cam.ac.uk/people/hg470">Hongyun Gao</a>, <a href="https://www.cl.cam.ac.uk/~aco41/">Cengiz Oztireli</a></p>

<p align="center">Department of Computer Science and Technology<br>University of Cambridge</p>

<p align="center"><a href="https://chenliang-zhou.github.io/FrePolad/">[Project page]</a>      <a href="https://arxiv.org/abs/2311.12090">[Paper]</a></p>

![teaser](docs/img/teaser.png)


# Abstract
We propose *FrePolad: **f**requency-**re**ctified **po**int **la**tent **d**iffusion*, a point cloud generation pipeline integrating a variational autoencoder (VAE) with a denoising diffusion probabilistic model (DDPM) for the latent distribution. FrePolad simultaneously achieves high quality, diversity, and flexibility in point cloud cardinality for generation tasks while maintaining high computational efficiency. The improvement in generation quality and diversity is achieved through (1) a novel frequency rectification via spherical harmonics designed to retain high-frequency content while learning the point cloud distribution; and (2) a latent DDPM to learn the regularized yet complex latent distribution. In addition, FrePolad supports variable point cloud cardinality by formulating the sampling of points as conditional distributions over a latent shape distribution. Finally, the low-dimensional latent space encoded by the VAE contributes to FrePolad’s fast and scalable sampling. Our quantitative and qualitative results demonstrate FrePolad’s state-of-the-art performance in terms of quality, diversity, and computational efficiency.

# Usage
The experiment can be run by
```
python train.py [OPTIONS]
```

Please refer to
```
python train.py --help
```
for help in passing the arguments.

# Citation
```
@inproceedings{zhou2023frepolad,
  title={FrePolad: Frequency-Rectified Point Latent Diffusion for Point Cloud Generation},
  author={Zhou, Chenliang and Zhong, Fangcheng and Hanji, Param and Guo, Zhilin and Fogarty, Kyle and Sztrajman, Alejandro and Gao, Hongyun and Oztireli, Cengiz},
  journal={ECCV 2024},
  year={2024}
}
```
