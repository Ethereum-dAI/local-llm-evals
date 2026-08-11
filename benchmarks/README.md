# General capability benchmarks

We fine-tuned gemma-4-E4B-it for wallet tool calling. The task accuracy is evaluated
elsewhere in this repo. This folder answers a different question: did the fine-tune
make the model worse at everything else? And since we ship a quantized GGUF, what does
Q4_K_M quantization cost on top?

We ran four variants of the model through the same six benchmarks with identical
prompts, seed and settings, using EleutherAI's lm-evaluation-harness (0.4.12).

## The four variants

| Name | What it is |
|---|---|
| gemma-4-E4B-it | the instruction-tuned base from Google, bf16 |
| gemma-4-E4B-wallet-ft | base plus our LoRA adapter (r=16), bf16 |
| gemma-4-E4B-it Q4_K_M | the base, converted and quantized by us with llama.cpp |
| gemma-4-E4B-wallet-ft Q4_K_M | the GGUF we actually ship |

The adapter was trained on the it model, so that is the baseline. The raw pretrained
gemma-4-E4B would be the wrong comparison. Google does not publish a quantized version
of the base, so we produced one ourselves at the same quant type to isolate the
quantization effect.

## Results

Metrics: acc for MMLU, Winogrande and TruthfulQA (mc2), acc_norm for HellaSwag and
ARC-Challenge, strict exact match for GSM8K. The ± values are standard errors.

| Benchmark | Shots | gemma-4-E4B-it | wallet-ft | it Q4_K_M | wallet-ft Q4_K_M |
|---|---|---|---|---|---|
| MMLU | 5 | 62.4 ±0.4 | 66.9 ±0.4 | 61.4 ±0.4 | 64.7 ±0.4 |
| GSM8K | 5 | 83.9 ±1.0 | 82.7 ±1.0 | 79.6 ±1.1 | 78.1 ±1.1 |
| HellaSwag | 0 | 68.1 ±0.5 | 68.7 ±0.5 | 66.6 ±0.5 | 66.9 ±0.5 |
| ARC-Challenge | 0 | 56.8 ±1.4 | 56.1 ±1.5 | 55.6 ±1.5 | 54.7 ±1.5 |
| Winogrande | 0 | 67.0 ±1.3 | 66.4 ±1.3 | 63.7 ±1.4 | 63.5 ±1.4 |
| TruthfulQA | 0 | 59.1 ±1.6 | 57.0 ±1.6 | 57.8 ±1.6 | 55.2 ±1.6 |

## What we take from this

The fine-tune did not damage the model. Four of the six benchmarks moved less than one
point, which is inside noise. MMLU actually improved by 4.5 points. The only score that
went the wrong way is TruthfulQA, down 2.1 points, roughly one standard error on the
noisiest benchmark of the set. Worth a second look on the next training run, not a
problem today.

Quantization costs between 1 and 4.6 points and hits GSM8K and Winogrande hardest.
The losses are practically identical on the base and on the fine-tune, so the shipped
GGUF behaves like the fine-tune plus ordinary 4-bit loss. Nothing compounds.

One caveat when comparing these numbers to Google's published results: Google evaluates
Gemma 4 on a newer benchmark set (MMLU-Pro, GPQA, AIME) with thinking mode enabled.
Our runs are plain loglikelihood and generation evals without thinking, so our absolute
numbers sit lower by construction. The comparison between our four variants is what
matters here, and all four ran under identical conditions.

## Reproducing

The bf16 runs use the harness HF backend:

```bash
lm_eval --model hf \
  --model_args "pretrained=google/gemma-4-E4B-it,dtype=bfloat16,add_bos_token=True" \
  --tasks hellaswag,arc_challenge,winogrande,truthfulqa_mc2 --num_fewshot 0 \
  --batch_size auto --seed 1234
# add peft=<adapter dir> to the model_args for the fine-tune
# mmlu and gsm8k run the same way with --num_fewshot 5
```

`add_bos_token=True` is not optional. gemma-4 tokenizers ship with add_bos_token set to
false and the harness does not know the gemma4 model type, so without the flag every
benchmark scores at chance level. We learned this the hard way.

The quantized runs use `scripts/eval_gguf_inproc.py`, which loads the GGUF in process
with llama-cpp-python and plugs into the harness Python API:

```bash
python scripts/eval_gguf_inproc.py --model model.Q4_K_M.gguf \
  --label llk0 --fewshot 0 --tasks hellaswag,arc_challenge,winogrande,truthfulqa_mc2
```

We wrote this after rejecting two other routes. Loading GGUFs through transformers
produces garbage logits for the gemma4 architecture without any error, and the harness
gguf HTTP connector truncates generative tasks at 16 tokens, has no request timeout and
spends 7 to 19 seconds per request. Build llama-cpp-python with CUDA
(`CMAKE_ARGS=-DGGML_CUDA=on`, pip installs a CPU build by default) and import torch
before llama_cpp. `scripts/convert_base.sh` shows how we produced the quantized base.

Raw harness outputs are in `results/`. The MMLU runs for the quantized models were
split in two halves (files ending in mmluA and mmluB) and merged as a sample-weighted
average, which is the same aggregation the harness uses. `results/summary_table.json`
holds the final table above in machine-readable form.
