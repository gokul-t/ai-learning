# Pytorch ROCm Installation Guide for AMD GPUs

```sh
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
```
