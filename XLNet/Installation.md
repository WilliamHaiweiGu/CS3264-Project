# Install the Environment for XLNet on Windows, Linux, or MacOS.

1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) according to your OS.
2. Activate Conda if the command line does not start with `(base)`.
3. Create a Conda environment with Python 3.12 and activate it.
```
conda create -n XLNet python=3.12 -y && conda activate XLNet
```
4. Install [PyTorch](https://pytorch.org/get-started/locally/)  according to your OS and GPU.
5. Install `requirements.txt`.
```
pip install -r requirements.txt
```