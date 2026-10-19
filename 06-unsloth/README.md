# PyTorch ROCm setup for AMD GPUs

Run these commands from `06-unsloth`. The project is configured to resolve
PyTorch and torchvision from the ROCm 7.14 wheel index.

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export MIOPEN_DEBUG_CONV_IMMED_FIND=0
export MIOPEN_DEBUG_DISABLE_ASM_KERNELS=1

mkdir -p "$HOME/uv_large_tmp/cache"
export TMPDIR="$HOME/uv_large_tmp"
export TEMP="$HOME/uv_large_tmp"
export TMP="$HOME/uv_large_tmp"
export UV_CACHE_DIR="$HOME/uv_large_tmp/cache"

uv sync
uv run python training.py
```
