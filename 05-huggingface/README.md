# Pytorch ROCm Installation Guide for AMD GPUs

```sh
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export MIOPEN_DEBUG_CONV_IMMED_FIND=0
export MIOPEN_DEBUG_DISABLE_ASM_KERNELS=1

# 1. Create a dedicated folder inside your large home partition
mkdir -p $HOME/uv_large_tmp 

# 2. Re-route the standard Linux temporary environment variables
export TMPDIR=$HOME/uv_large_tmp 
export TEMP=$HOME/uv_large_tmp 
export TMP=$HOME/uv_large_tmp 
export UV_CACHE_DIR=$HOME/uv_large_tmp/cache

# 3. Re-route uv's internal cache framework explicitly
rm -rf .venv
uv venv --system-site-packages .venv 
source .venv/bin/activate
uv pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/rocm7.14

# 4. Clean up custom staging area
rm -rf $HOME/uv_large_tmp

# 5. Run the python program
uv run python generate.py
```

```

