# MNNPDE
Multiscale neural network for PDEs

## MNN
Multiscale neural network based on
- [x] H-matrices
- [x] FMM / H^2-matrices
- [x] BCR-algorithm

## Application
- [ ] operator (Jordi)
- [ ] inverse problem
- [ ] imagenet



## Data expectations

h5 file with keys 'measure' and 'coe'. 

The 'measure' key has shape (N_samples, N_s, N_d)
The 'coe' key has shape (N_samples, N_t, N_r)

