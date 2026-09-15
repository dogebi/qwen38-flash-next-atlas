# Qwen3.8-Flash-Next — Tensor Atlas v4

An offline, single-file WebGL2 architecture atlas for **[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**
(Qwen Community License 1.0), built with the `tensor-atlas-v4-retarget` skill: nisten's Tensor Atlas v4
engine kept whole, the model half replaced. `index.html` is self-contained — no CDN, no runtime requests.

Live: https://qwen38-flash-next-atlas.netlify.app

## What this model is

The experimental preview of the architecture that will underpin Qwen4: **125B parameters with 6B activated,
plus a 51B n-gram embedding and a 4B MTP layer.** Its 48 layers are laid out as
`12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))`:

* **Gated DeltaNet** (36 layers) — constant-state linear attention: 48 value heads / 16 query-key heads at
  head_dim 128, a 4-tap convolution over the delta-rule state, learned per-head decays and an output gate.
* **Qwen Sparse Attention** (12 layers) — 24 query heads, 2 KV heads at head_dim 256 (64 of them rotary),
  with a micro-block **indexer** (MQA, 4 query heads, 1 shared key head) spending a 512-block / 2048-token budget.
* **Gated Residual** — four hyper-connection tensors around the attention block and four around the MoE block
  in every layer (4 branches, bottleneck rank 320), plus a final mixer.
* **MoE** — 512 routed experts (top-10, sigmoid routing) at intermediate 640, plus one always-on shared expert.
* **N-gram embedding** — 20,000,000 bigram/trigram rows in 128 shards at layer 2, an axis of parameter scaling
  the card describes as easier to offload than experts.
* **Vision tower** (27 blocks) and **one MTP layer** (multi-token prediction).
* Context: 262,144 tokens natively, extensible to 1,000,000.

## Measured payloads

Every byte is the sum of tensor byte ranges read from the published safetensors **headers** over HTTP Range —
no weights were downloaded. Quantisation scales (`weight_scale`, `weight_scale_inv`), packed weights
(`weight_packed`) and per-expert tensors are folded onto the logical tensor they belong to, so all three modes
list the same 1,530 logical tensors.

| mode | repository | shards | measured payload |
|---|---|---|---|
| BF16 | Qwen/Qwen3.8-Flash-Next | 131 | 360.0000 GB |
| FP8 | Qwen/Qwen3.8-Flash-Next-FP8 | 131 | 185.5022 GB |
| NVFP4 | nvidia/Qwen3.8-Flash-Next-NVFP4 | 11 | 132.6398 GB |

Notable: the FP8 build leaves 943 module patterns unquantised (norms, routers, gates, most hyper-connection
tensors, the vision tower) while explicitly converting the n-gram embedding; the NVFP4 build has *more*
tensors than the reference (per-block scale + global scale per weight) and the smallest payload.

## Build

```bash
python3 measure_fast.py Qwen/Qwen3.8-Flash-Next Qwen/Qwen3.8-Flash-Next-FP8 nvidia/Qwen3.8-Flash-Next-NVFP4
python3 build.py                # gates + preflight -> index.html
python3 preflight.py --selftest # prove the checker can still fail
```

`build.py` writes `index.html` only when the page's declared totals equal the measured payload of **all three**
modes, the expected copy is present and no earlier model's copy survives, the preflight self-test passes, and
the candidate passes every structural check that applies (payload, palette keys, `pos()` targets, id loops,
`node --check` on the inline script, the tick guard, the app shell, the storage layout).

## Layout

```
index.html          the deliverable — open it from disk, it needs nothing
src.html            nisten's Tensor Atlas v4 engine (MIT, (c) 2026 netsin), unmodified
build.py            model half + gates: folds shard headers into logical tensors, then patches the engine
panels.py           panel copy and whole-line rewrites (anchored on the engine's own lines)
objectcode.py       rewrite helpers (balance-checked line rewrites, literal substitutions)
preflight.py        the structural checker (run it after any edit)
measure.py / measure_fast.py   safetensors header reader, sequential / 10-way parallel
hf-*.json           measured headers for the three modes + the three published configs
bench.json          the model card's benchmark table (five models, as published)
```

## Credits

* Engine: nisten, **LLMViz-DeepSeek-V4.1-Flash** (MIT, © 2026 netsin) — reused unmodified.
* Architecture facts and benchmark figures: the Qwen3.8-Flash-Next model card (`README-model-card.md`).
* Weights: Qwen (BF16, FP8) and NVIDIA (NVFP4). Metadata only — no weight shards are redistributed.
* Model licence: Qwen Community License 1.0 (`LICENSE-MODEL.txt`).
