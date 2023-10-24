## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Getting started](#getting-started)

## General info
Working space for a research project exploring super-resolution of 2D turbulence.

## Technologies
The project was created with:
* python 3.9.13
* NumPy
* SciPy
* [jax 0.4.2]([JAX](https://github.com/google/jax))
* [jax-cfd 0.2.0](https://github.com/google/jax-cfd/tree/main)
* [fenics-dolfin 2019.1.0](https://github.com/FEniCS)
* [Oasis 2018](https://github.com/mikaem/Oasis)

## Setup

After the conda environment is created, e.g.,
```
conda create -n observer python=3.9
```
we then install the required packages. For data generation, we are using [JAX-CFD](https://github.com/google/jax-cfd/tree/main). For the application to work,
[JAX](https://github.com/google/jax) needs to be installed. A base install with pip install JAX-CFD only requires NumPy, SciPy and JAX.

To install a CPU-only version of JAX, which might be useful for doing local development on a laptop, you can run
```
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```
If you want to install JAX with NVIDIA GPU support, you can use CUDA and CUDNN installed from pip wheels; CUDA and CuDNN installation need to match the wheels available, e.g.,

```
pip install --upgrade pip

# Installs the wheel compatible with CUDA 11 and cuDNN 8.6 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
More details are available in the JAX documentation.

To locally install JAX-CFD, do the following:

```
git clone https://github.com/google/jax-cfd.git
cd jax-cfd
pip install jaxlib
pip install -e ".[complete]"
```

To install FEniCS follow the instructions in the [documentation](https://github.com/FEniCS/dolfinx#installation).

Oasis is installed with regular distutils
```
git clone https://github.com/mikaem/Oasis.git
cd Oasis
python setup.py install --prefix='Path to where you want Oasis installed. Must be on PYTHONPATH'
```
We are assuming that the installation is in the current work directory, i.e., './'.


To be able to run the CFD solver, we need [Oasis](https://github.com/mikaem/Oasis) and a compatible installation of [FEniCS](https://fenicsproject.org/).
After downloading Oasis, apply a patch provided to a file in the Oasis folder:
```
patch -u ./Oasis/oasis/NSfracStep.py -i NSfracStep.patch
```
## Getting started
If not present, create two empty folders: `data` and `observer_results`.

The following steps will enable data generation. We simulate 2D Kolmogorov flow for a given set of parameters. Executing:

```
mkdir data/r123_g256by256_nu0.006
python kolmogorov_flow.py 123 512 0.006 data/r123_g256by256_nu0.006
```
will produce simulation data that were used in our study with the grid resolution of 512x512, viscosity set to 0.006 and 123 being a random number used for the initial field construction. The results will be stored in the defined folder `data/r123_g256by256_nu0.006`.

You can also just take advantage of the script for multiple case generation `generate_observations.sh`. Specify settings inside and execute:

```
./generate_observations.sh
```
We also provide a script `NoiseForwardTemplate_mpi.py` which enables data generation using FEniCS.
To run the experiments with the observer, use the script provided to set up a number of different cases:
```
./generate_cases_and_run_study.sh
```

Inside the script, provide the parameters for the study:
```
WORK_DIR=$PWD/observer_results
NPROCS=2
grids=( 64)
crs=( 1 2 4 8 16)
alphas=( 0.2 0.4 0.8)
gains=( 5)
seeds=($(seq 123 1 123))
nu=0.006
```
You can specify the number of processors for MPI run, the grid size, compression ratios (crs), the amount of noise added (alphas), viscosity, and the gain of the observer.

If this script is run on a MAC, add `''` to the lines with `sed`, i.e., replace
```
sed -i "s/gridflag/$g/" NoiseLuenberger_mpi.py
```
with
```
sed -i '' "s/gridflag/$g/" NoiseLuenberger_mpi.py
```
Otherwise, remove.
