#!/usr/bin/env bash
#
# Reproducible conda env for nano-vllm on a Meta GPU devserver (H100).
# VERIFIED WORKING 2026-06-15: ~13,800 tok/s on one H100 with Qwen3-0.6B.
#
# WHY THIS IS NOT THE README's one-line `pip install`:
#   The devserver reaches the internet only through fwdproxy, whose allowlist
#   permits ONLY: pypi.org / files.pythonhosted.org, repo.anaconda.com (conda
#   defaults), download.pytorch.org, and huggingface.co.  It BLOCKS (403):
#     - GitHub releases, the nvidia conda channel, 3rd-party pip registries
#       (=> no prebuilt flash-attn wheel; must COMPILE from source)
#     - pypi.nvidia.com (=> torch's pinned nvidia deps, normally fetched via the
#       cu128 index, must come from REGULAR PyPI with EXACT version pins)
#     - us.aws.cdn.hf.co  (HF Xet CDN => must disable Xet for model download)
#
# PROVEN VERSION TRIPLE (CUDA 12.8 / torch 2.7.0+cu128 / flash-attn 2.7.4.post1),
# python 3.12.  See VERSIONS_EXPLAINED.md for why these must all match.
# NOTE: transformers must be >=4.56 (repo HEAD reads config.dtype, renamed from
# torch_dtype in 4.56); the older 4.51.3 fails with AttributeError.
#
# OTHER GOTCHAS THIS SCRIPT WORKS AROUND (all hit during bring-up):
#   - A corrupted pip cache produced half-extracted packages -> --no-cache-dir
#     everywhere + `pip cache purge`.
#   - conda's cuda-toolkit puts headers in $ENV/targets/x86_64-linux/include,
#     NOT $ENV/include -> export CPATH so the flash-attn .cpp build finds them.
#   - flash-attn defaults to many GPU archs -> FLASH_ATTN_CUDA_ARCHS=90 builds
#     H100-only (sm_90) and roughly halves compile time.
#   - At RUNTIME you must export LD_LIBRARY_PATH=$ENV/targets/.../lib:$ENV/lib
#     so torch/flash-attn find libcudnn etc.
#
# Usage:  bash setup_h100.sh
set -euo pipefail

# ---- config (override via env) ----
ENV_NAME="${ENV_NAME:-nanovllm}"
PY_VERSION="${PY_VERSION:-3.12}"
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
TORCH_VER="${TORCH_VER:-2.7.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
CUDNN_VER="${CUDNN_VER:-9.7.1.26}"
FA_VER="${FA_VER:-2.7.4.post1}"
FA_ARCHS="${FA_ARCHS:-90}"            # 90=H100 only; use "80;90" for A100+H100
MAX_JOBS="${MAX_JOBS:-64}"
REPO_DIR="${REPO_DIR:-/home/cicichen/nano-vllm}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"
MODEL_DIR="${MODEL_DIR:-$HOME/huggingface/$(basename "$MODEL_ID")}"
CONDA="${CONDA:-/usr/bin/conda}"     # real binary; conda shell fn isn't loaded non-interactively

# torch 2.7.0+cu128's EXACT pinned nvidia deps (from `pip show`/wheel metadata).
# These exact versions are on regular PyPI; unversioned would pull 12.9.x (soname
# mismatch -> libcusparseLt.so.0 not found).
NVIDIA_PINS=(
  nvidia-cublas-cu12==12.8.3.14 nvidia-cuda-cupti-cu12==12.8.57
  nvidia-cuda-nvrtc-cu12==12.8.61 nvidia-cuda-runtime-cu12==12.8.57
  nvidia-cufft-cu12==11.3.3.41 nvidia-cufile-cu12==1.13.0.11
  nvidia-curand-cu12==10.3.9.55 nvidia-cusolver-cu12==11.7.2.55
  nvidia-cusparse-cu12==12.5.7.53 nvidia-cusparselt-cu12==0.6.3
  nvidia-nccl-cu12==2.26.2 nvidia-nvjitlink-cu12==12.8.61 nvidia-nvtx-cu12==12.8.55
)

# ---- proxy ----
export https_proxy=http://fwdproxy:8080  http_proxy=http://fwdproxy:8080
export HTTPS_PROXY=http://fwdproxy:8080  HTTP_PROXY=http://fwdproxy:8080

echo "== 1. create a TRULY CLEAN conda env '$ENV_NAME' (python $PY_VERSION) =="
# rm -rf the prefix first: `conda env remove` can leave a polluted prefix behind,
# and a leftover ML stack with half-broken packages will poison imports.
"$CONDA" env remove -n "$ENV_NAME" -y 2>/dev/null || true
rm -rf "$("$CONDA" info --base)/envs/$ENV_NAME" "$HOME/.conda/envs/$ENV_NAME" 2>/dev/null || true
"$CONDA" create -n "$ENV_NAME" "python=$PY_VERSION" -y
ENV_PREFIX="$("$CONDA" env list | awk -v n="$ENV_NAME" '$NF ~ ("/" n "$"){print $NF}')"
PY="$ENV_PREFIX/bin/python"
echo "   prefix: $ENV_PREFIX"
SP_COUNT="$(ls "$ENV_PREFIX/lib/python$PY_VERSION/site-packages" | wc -l)"
echo "   site-packages entries: $SP_COUNT (expect ~10-15; if hundreds, the prefix wasn't clean)"
"$PY" -m pip cache purge 2>/dev/null || true

echo "== 2. CUDA $CUDA_VERSION toolkit (nvcc + headers) from reachable 'defaults' channel =="
"$CONDA" install -n "$ENV_NAME" -c defaults "cuda-toolkit=$CUDA_VERSION" -y
"$ENV_PREFIX/bin/nvcc" --version | tail -1

echo "== 3. torch + its deps (work around blocked pypi.nvidia.com) =="
# 3a. torch wheel itself is on download.pytorch.org (allowed); --no-deps avoids
#     pip resolving nvidia deps through the cu128 index (which routes to the
#     blocked pypi.nvidia.com).
"$PY" -m pip install "torch==$TORCH_VER" --index-url "$TORCH_INDEX" --no-deps --no-cache-dir
# 3b. cudnn + the exact nvidia libs from REGULAR PyPI:
"$PY" -m pip install "nvidia-cudnn-cu12==$CUDNN_VER" "${NVIDIA_PINS[@]}" --no-cache-dir
# 3c. torch's pure-python deps:
"$PY" -m pip install "numpy<2.3" sympy networkx jinja2 filelock fsspec typing-extensions \
    triton==3.3.0 einops --no-cache-dir

echo "== 4. verify torch sees the GPU =="
env -u PYTHONPATH "$PY" -c "import torch; assert torch.cuda.is_available(); print('CUDA ok ->', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| cuda', torch.version.cuda)"

echo "== 5. compile flash-attn $FA_VER from source (H100/sm_$FA_ARCHS only; ~10-25 min) =="
"$PY" -m pip install ninja psutil packaging --no-cache-dir
export CUDA_HOME="$ENV_PREFIX"
export CPATH="$ENV_PREFIX/targets/x86_64-linux/include"
export LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib"
export LD_LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib"
export FLASH_ATTN_CUDA_ARCHS="$FA_ARCHS" TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS="$MAX_JOBS"
"$PY" -m pip install "flash-attn==$FA_VER" --no-build-isolation --no-deps --no-cache-dir

echo "== 6. transformers (>=4.56 for config.dtype) + nano-vllm =="
"$PY" -m pip install $TRANSFORMERS_SPEC tokenizers "huggingface_hub<1.0" xxhash \
    safetensors regex pyyaml requests tqdm --no-cache-dir
"$PY" -m pip install --no-deps -e "$REPO_DIR"

echo "== 7. verify the full import chain =="
env -u PYTHONPATH LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$PY" -c "import torch, flash_attn, nanovllm, xxhash; from transformers import AutoTokenizer, AutoConfig; from nanovllm import LLM, SamplingParams; print('imports OK | flash_attn', flash_attn.__version__)"

echo "== 8. download model: $MODEL_ID -> $MODEL_DIR  (Xet CDN is blocked -> disable it) =="
HF_HUB_DISABLE_XET=1 "$ENV_PREFIX/bin/hf" download "$MODEL_ID" --local-dir "$MODEL_DIR"

echo "== 9. smoke test =="
MODEL="$MODEL_DIR" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" env -u PYTHONPATH "$PY" "$REPO_DIR/test_h100.py"

echo
echo "== DONE =="
echo "To run later, FIRST export the runtime lib path:"
echo "  export LD_LIBRARY_PATH=$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib"
echo "  MODEL=$MODEL_DIR $PY $REPO_DIR/test_h100.py"
