```bash
conda create -n rlinf-recap310 python=3.10 pip -y
conda activate rlinf-recap310

export RLINF_SRC=/data/workspaces/RLinf
cd "$RLINF_SRC"

python -m pip install -U pip uv wheel tomli "setuptools==75.8.0"

python --version
which python
nvidia-smi


```