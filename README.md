# Monte Carlo Maximum Likelihood Reconstruction for Digital Holography with Speckle

## Preview
#### There are 5 python files in this repo. The test data is under the /data/ folder.

- train_PGD_MC_complex.py: training PGD-MC algorithm for recovering speckle-free real-valued reflectivity image from complex-valued holographic measurements with speckle.

- PGD_MC_complex.py: implementation of Monte-Carlo sampling and conjugate gradient methods for matrix-free maximum likelihood based reconstruction for digital holography.

- decoder.py: basic network structures of the Deep Decoder we use for projection.

- train_DnCNN_origin.py: train DnCNN denoiser as prior model.

- utils.py: all the other helper functions.

## Run the simulation

#### Run the PGD-MC algorithm (efficient Monte-Carlo and conjugate gradient methods) for recovering images from holographic measurements with speckle:

```
python train_PGD_MC_complex.py
```

#### Specify the hyperparameters and experiment setting:

#### E.g., recover images from measurements with number of looks L=1, circular aperture radius ratio=1.0, additive noise level=25, Monte-Carlo samples=10, denoiser=DIP:

```
python train_PGD_MC_complex.py --dataset 'peppers' --mask_rate 1.0 --num_look 1 --add_std 0.2 --add_std_prime 0.2 --lr_GD 0.01 --outer_ite 100 --num_ite_MC 10 --denoiser 'DIP' --lr_NN 1e-3 --inner_ite 200
```

## Relevant works on image reconstruction in coherent imaging with speckle

[1] Chen, Xi, Soham Jana, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Multilook Coherent Imaging: Theoretical Guarantees and Algorithms." arXiv preprint arXiv:2505.23594 (2025) [paper](https://arxiv.org/pdf/2505.23594)

[2] Chen, Xi, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Chen, Xi, et al. "Efficient multilook coherent imaging with temporally dependent speckle noise." Unconventional Imaging, Sensing, and Adaptive Optics 2025. Vol. 13619. SPIE, 2025. [paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13619/1361915/Efficient-multilook-coherent-imaging-with-temporally-dependent-speckle-noise/10.1117/12.3063994.full)

[3] Chen, Xi, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Monte-Carlo Based Efficient Image Reconstruction in Coherent Imaging With Speckle Noise." 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI). IEEE, 2025. [paper](https://ieeexplore.ieee.org/abstract/document/10981291)

[4] Chen, Xi, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Novel approach to coherent imaging in the presence of speckle noise." Unconventional Imaging, Sensing, and Adaptive Optics 2024. Vol. 13149. SPIE, 2024. [paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13149/1314908/Novel-approach-to-coherent-imaging-in-the-presence-of-speckle/10.1117/12.3027824.full)

[5] Chen, Xi, Zhewen Hou, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Bagged Deep Image Prior for Recovering Images in the Presence of Speckle Noise." Forty-first International Conference on Machine Learning (ICML 2024). [paper](https://openreview.net/pdf?id=IoUOhnCmlX)

[6] Chen, Xi, Zhewen Hou, Christopher Metzler, Arian Maleki, and Shirin Jalali. "Multilook compressive sensing in the presence of speckle noise." In NeurIPS 2023 Workshop on Deep Learning and Inverse Problems. 2023. [paper](https://openreview.net/forum?id=G8wMnihF6E)

[7] Zhou, Wenda, Shirin Jalali, and Arian Maleki. "Compressed sensing in the presence of speckle noise." IEEE Transactions on Information Theory 68.10 (2022): 6964-6980. [paper](https://ieeexplore.ieee.org/abstract/document/9783054)
