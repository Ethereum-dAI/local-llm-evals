#!/bin/bash
# Convert google/gemma-4-E4B-it to Q4_K_M GGUF (same quant type as the FT artifact)
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/workspace/hf
cd /workspace
pip install -q -U gguf 2>&1 | tail -1
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp 2>&1 | tail -1
echo "=== download base ==="
hf download google/gemma-4-E4B-it --local-dir /workspace/base-hf 2>&1 | tail -1
echo "=== convert to bf16 gguf ==="
python llama.cpp/convert_hf_to_gguf.py /workspace/base-hf --outfile /workspace/base-bf16.gguf --outtype bf16 2>&1 | tail -3
echo "=== build llama-quantize (CPU) ==="
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF > /dev/null
cmake --build llama.cpp/build --target llama-quantize -j"$(nproc)" 2>&1 | tail -2
echo "=== quantize Q4_K_M ==="
llama.cpp/build/bin/llama-quantize /workspace/base-bf16.gguf /workspace/base.Q4_K_M.gguf Q4_K_M 2>&1 | tail -3
rm -f /workspace/base-bf16.gguf
ls -la /workspace/base.Q4_K_M.gguf
echo CONVERT_DONE
