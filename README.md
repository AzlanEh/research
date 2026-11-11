# wifi-extend

## Python environment

This project uses a **self-contained conda environment** stored at `./.venv`.
It does not touch conda `base` or any other project's environment.

- Python: 3.11
- Location: `/home/iotlabnew/workspace/wifi-extend/.venv`

### Activate

```bash
conda activate /home/iotlabnew/workspace/wifi-extend/.venv
```

A conda env has no `bin/activate` script. If conda isn't initialized in your
shell (e.g. a non-interactive script), load the conda hook first:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/iotlabnew/workspace/wifi-extend/.venv
```

Or skip activation entirely by using the binaries directly:

```bash
./.venv/bin/python script.py
./.venv/bin/pip install <package>
```

Check you are in the right env:

```bash
which python        # → /home/iotlabnew/workspace/wifi-extend/.venv/bin/python
```

Deactivate with `conda deactivate`.

### Install packages

Always use the project env explicitly:

```bash
.venv/bin/pip install -r requirements.txt
```

For PyTorch with CUDA 12.4 (RTX A6000, driver 550):

```bash
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> Never run a bare `pip install` while conda `base` is active — that would
> modify the shared base environment.

### Recreate / reproduce

```bash
conda create -p ./.venv python=3.11 -y
.venv/bin/pip install -r requirements.txt
```

Or export the current env:

```bash
conda env export -p .venv --from-history > environment.yml
```
