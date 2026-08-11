#!/usr/bin/env python3
"""In-process GGUF eval for lm-eval-harness via llama-cpp-python.

Avoids the HTTP server's echo+logprobs path (which builds top-10 logprob dicts
for every prompt token — ~7s/request). Computes only continuation-token
logprobs directly from the logits buffer.

Usage: eval_gguf_inproc.py --label llk0 --fewshot 0 --tasks hellaswag,arc_challenge ...
"""
import argparse, json, os, sys
import torch  # noqa: F401 — must load before llama_cpp (NCCL symbol clash)
import numpy as np
from tqdm import tqdm
from llama_cpp import Llama
from lm_eval import simple_evaluate
from lm_eval.api.model import LM

GGUF = "/workspace/gguf/gemma-4-E4B-wallet-ft.Q4_K_M.gguf"


def log_softmax_row(row):
    m = row.max()
    return row - m - np.log(np.exp(row - m).sum())


class LlamaCppInproc(LM):
    def __init__(self, path=GGUF, n_ctx=4096):
        super().__init__()
        self.llm = Llama(path, n_gpu_layers=-1, n_ctx=n_ctx, logits_all=True, verbose=False)

    def _tok(self, s):
        return self.llm.tokenize(s.encode("utf-8"), add_bos=True, special=False)

    def loglikelihood(self, requests, disable_tqdm=False):
        res = []
        for ctx, cont in tqdm([r.args for r in requests], disable=disable_tqdm, mininterval=30):
            full = self._tok(ctx + cont)
            ctx_toks = self._tok(ctx)
            # boundary = longest common prefix (handles tokens merging across the seam)
            n = 0
            for a, b in zip(full, ctx_toks):
                if a != b:
                    break
                n += 1
            n = max(n, 1)  # always condition on at least BOS
            if len(full) > self.llm.n_ctx():
                full = full[-self.llm.n_ctx():]
                n = max(n - (len(full) - self.llm.n_ctx()), 1)
            self.llm.reset()
            self.llm.eval(full)
            scores = self.llm.scores  # (n_ctx, n_vocab) float32, row i predicts token i+1
            total, greedy = 0.0, True
            for j in range(n, len(full)):
                lp = log_softmax_row(scores[j - 1].astype(np.float64))
                total += lp[full[j]]
                if int(np.argmax(scores[j - 1])) != full[j]:
                    greedy = False
            res.append((float(total), greedy))
        return res

    def generate_until(self, requests, disable_tqdm=False):
        res = []
        for req in tqdm([r.args for r in requests], disable=disable_tqdm, mininterval=30):
            prompt, kwargs = req[0], req[1]
            until = kwargs.get("until", ["</s>"])
            out = self.llm.create_completion(
                prompt, max_tokens=512, stop=until, temperature=0.0)
            res.append(out["choices"][0]["text"])
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--fewshot", type=int, required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    lm = LlamaCppInproc()
    out = simple_evaluate(
        model=lm,
        tasks=args.tasks.split(","),
        num_fewshot=args.fewshot,
        limit=args.limit,
        random_seed=1234, numpy_random_seed=1234, fewshot_random_seed=1234,
    )
    os.makedirs("/workspace/results", exist_ok=True)
    path = f"/workspace/results/gguf_{args.label}.json"
    with open(path, "w") as f:
        json.dump({"results": out["results"], "n-samples": out.get("n-samples")}, f, default=str)
    for t, m in out["results"].items():
        print(t, {k: v for k, v in m.items() if isinstance(v, (int, float))})
    print("WROTE", path)


if __name__ == "__main__":
    main()
