# DGM TEAM6 DiffuseVAE CODE

This repo contains the official implementation of the paper: [DiffuseVAE: Efficient, Controllable and High-Fidelity Generation from Low-Dimensional Latents](https://arxiv.org/abs/2201.00308) by [Kushagra Pandey](https://kpandey008.github.io/), [Avideep Mukherjee](https://www.cse.iitk.ac.in/users/avideep/), [Piyush Rai](https://www.cse.iitk.ac.in/users/piyush/), [Abhishek Kumar](http://www.abhishek.umiacs.io/)
and joint end-to-end , adaptive conditioning, adative condition with flow model

## TODO Before Test or Train 
Update the `chkpt_path` and `root directory` in the train and test YAML files located in `configs/dataset/cifar10/*.yaml`.

We uploaded every chkpt files and sampling results images in [here](https://drive.google.com/file/d/1rFWcEZ8DBmJRUPa3nKmQLIKRgDESfHeJ/view?usp=drive_link).

## Train Code

1. Joint end to end DiffuseVAE 
```
python train_joint_end_to_end.py
``` 

2. Adaptive conditioning DiffuseVAE
```
python train_adaptive_cond.py
``` 

3. Adaptive condition with Flow model
```
python train_flow_ae.py
python train_flow_ddpm.py
``` 
## Test Code(sampling)

1. Joint end to end DiffuseVAE 
```
python sample_cond_joint_end_to_end.py
``` 

2. Adaptive conditioning DiffuseVAE
```
python sample_cond_adaptive_cond.py
``` 

3. Adaptive condition with Flow model
```
python sample_cond_flow_ddpm.py
``` 

# Sampling Results 2000epoch
Inception Score, Fréchet Inception Distance, 50K sampling Time
1. VANILLA DiffuseVAE
	IS  9.61890303069825 ± 0.1117982889566477
	FID 3.8238908881344855
	Time 1.16 hr

2. Joint end-to-end 
	IS  8.855085444074216 ± 0.12010746053058496
	FID 6.138252127239014
	time 2.43 hr
	
3. Adaptive Condition model
	IS  9.393495792280843 ± 0.10574670689662867
	FID 4.25968282086194
	time 1.48 hr

4. Adaptive Condition With Flow based model
	IS  9.53732209902916 ± 0.11355254137873153
	FID 4.737415015767681
	time 4.86 hr


## Citing
To cite DiffuseVAE please use the following BibTEX entries:

```
@misc{pandey2022diffusevae,
      title={DiffuseVAE: Efficient, Controllable and High-Fidelity Generation from Low-Dimensional Latents}, 
      author={Kushagra Pandey and Avideep Mukherjee and Piyush Rai and Abhishek Kumar},
      year={2022},
      eprint={2201.00308},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```
@inproceedings{
pandey2021vaes,
title={{VAE}s meet Diffusion Models: Efficient and High-Fidelity Generation},
author={Kushagra Pandey and Avideep Mukherjee and Piyush Rai and Abhishek Kumar},
booktitle={NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications},
year={2021},
url={https://openreview.net/forum?id=-J8dM4ed_92}
}
```

Since our model uses diffusion models please consider citing the original [Diffusion model](https://arxiv.org/abs/1503.03585), [DDPM](https://arxiv.org/abs/2006.11239) and [VAE](https://arxiv.org/abs/1312.6114) papers.
