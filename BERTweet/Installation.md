# Install the Environment for XLNet on Windows, Linux, or MacOS.

1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) according to your OS.
2. Activate Conda if the command line does not start with `(base)`.
3. Create a Conda environment with Python 3.12 and activate it.
```
conda create -n BERTweet python=3.12 -y && conda activate BERTweet
```
5. Install dependencies.
```
pip install -r requirements.txt
```
4. If you see an error msg saying `torch` is required, install [PyTorch](https://pytorch.org/get-started/locally/)  according to your OS and GPU.
