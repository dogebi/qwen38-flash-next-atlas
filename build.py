#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build qwen38-flash-next-atlas/index.html — skill: tensor-atlas-v4-retarget.

src.html is the nisten "Tensor Atlas v4" engine (LLMViz-DeepSeek-V4.1-Flash, MIT (c) 2026 netsin)
kept whole for the shell; only the model half is replaced.

Subject: Qwen/Qwen3.8-Flash-Next (Qwen Community License 1.0) — the experimental preview of the
architecture that will underpin Qwen4: 48 layers laid out as 12 x (3 x (Gated DeltaNet -> MoE) ->
1 x (Qwen Sparse Attention -> MoE)). Every layer also carries Gated Residual (hyper-connection)
mixers, the MoE is 512 routed experts (top-10) plus one shared expert, QSA layers add a micro-block
sparse indexer (budget 512 blocks / 2048 tokens), layer 1 owns a 20,000,000-entry n-gram embedding
table (128 shards), and there is one MTP layer plus a 27-block vision tower.

Three measured modes, per-tensor bytes read from the safetensors headers over HTTP Range:
  bf16   Qwen/Qwen3.8-Flash-Next          131 shards  1658 tensors  360.0002 GB
  fp8    Qwen/Qwen3.8-Flash-Next-FP8      131 shards           -     185.5233 GB
  nvfp4  nvidia/Qwen3.8-Flash-Next-NVFP4   11 shards           -     132.6802 GB
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
SRC, OUT = DIR / "src.html", DIR / "index.html"
BF_JSON, FP8_JSON, NV_JSON = DIR / "hf-bf16.json", DIR / "hf-fp8.json", DIR / "hf-nvfp4.json"
CFG_BASE, CFG_FP8, CFG_NV = DIR / "hf-config-bf16.json", DIR / "hf-config-fp8.json", DIR / "hf-config-nvfp4.json"

SCALE_SUFFIX = re.compile(
    r"\.(weight_scale_inv|weight_scale_2|weight_scale|weight_global_scale|weight_packed|scales|biases|input_scale|output_scale)$")
PREFIX = [(r"^model\.language_model\.", "model."), (r"^model\.visual\.", "visual.")]
NGRAM_SHARD = re.compile(r"\.ngram_embedding\.shard_\d+\.weight$")
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# the FP8 build keeps experts one by one; the reference publishes two fused banks. Fold them.
PER_EXPERT = re.compile(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)")


def logical(name: str) -> str:
    """Normalise a raw checkpoint name to the atlas's logical name (prefix + expert + n-gram folding)."""
    n = name
    for pat, rep in PREFIX:
        n = re.sub(pat, rep, n)
    if PER_EXPERT.search(n):
        which = PER_EXPERT.search(n).group(1)
        n = PER_EXPERT.sub(".mlp.experts." + ("down_proj" if which == "down_proj" else "gate_up_proj"), n)
    if NGRAM_SHARD.search(n):
        n = NGRAM_SHARD.sub(".ngram_embedding.weight", n)
    return n


def make_folder(bfset: set[str]):
    """Map any raw name onto a logical name the BF16 reference defines (scales onto their owner)."""
    def folder(name: str) -> str | None:
        n = logical(name)
        stripped = SCALE_SUFFIX.sub("", n)
        base = re.sub(r"\.weight$", "", stripped)
        for c in (n, stripped, base, base + ".weight", stripped + ".weight", n + ".weight"):
            if c in bfset:
                return c
        return None
    return folder


def collect(path: Path, folder) -> tuple[dict, list[str]]:
    out: dict[str, int] = {}
    unmapped: list[str] = []
    for n, t in load(path)["tensors"].items():
        lg = folder(n)
        if lg is None:
            unmapped.append(n)
            continue
        out[lg] = out.get(lg, 0) + t["bytes"]
    return out, unmapped


def layer_types() -> list[str]:
    tc = load(CFG_BASE)["text_config"]
    return list(tc["layer_types"])


def templates() -> dict:
    bfset = {logical(n) for n in load(BF_JSON)["tensors"]}
    folder = make_folder(bfset)
    bf, un_bf = collect(BF_JSON, folder)
    fp8, un_fp8 = collect(FP8_JSON, folder) if FP8_JSON.exists() else ({}, [])
    nv, un_nv = collect(NV_JSON, folder) if NV_JSON.exists() else ({}, [])
    for label, un in (("bf16", un_bf), ("fp8", un_fp8), ("nvfp4", un_nv)):
        if un:
            raise SystemExit(f"{label}: {len(un)} tensors map to nothing, e.g. {un[:4]}")
    names = sorted(set(bf) | set(fp8) | set(nv))
    if os.environ.get("ATLAS_FOLD_REPORT"):
        from collections import Counter
        for label, path in (("bf16", BF_JSON), ("fp8", FP8_JSON), ("nvfp4", NV_JSON)):
            if not path.exists():
                continue
            c = Counter(logical(n) for n in load(path)["tensors"])
            top = c.most_common(3)
            print(f"  fold report {label}: {len(c):,} logical names · biggest groups {top}")

    shapes: dict[str, list[int]] = {}
    for src in (BF_JSON, FP8_JSON, NV_JSON):
        if not src.exists():
            continue
        for raw, t in load(src)["tensors"].items():
            lg = folder(raw)
            if lg is None or lg in shapes:
                continue
            if raw.endswith("_packed") or not re.search(
                    r"\.(weight|pe|bias|A_log|dt_bias|layer_multipliers|ngram_heads_offsets|ngram_heads_vocab_sizes)$", raw):
                continue
            if len(t["shape"]) <= 3:
                shapes[lg] = t["shape"]
    print(f"  logical tensors: {len(shapes):,}")

    def dims_of(n: str) -> list[int]:
        return shapes.get(n, [1])

    both = {n: {"dims": dims_of(n), "b16": bf.get(n, 0), "b8": fp8.get(n, 0), "b4": nv.get(n, 0)}
            for n in names}

    lt = layer_types()
    groups: dict[str, dict] = {}
    for i, kind in enumerate(lt):
        pre = f"model.layers.{i}."
        got = {n[len(pre):]: v for n, v in both.items() if n.startswith(pre)}
        if not got:
            raise SystemExit(f"layer {i} has no tensors")
        core = {k: v for k, v in got.items() if not k.startswith("ple.")}
        key = kind + "|" + json.dumps({k: v["dims"] for k, v in sorted(core.items())})
        g = groups.setdefault(key, {"kind": kind, "indices": [], "items": core})
        g["indices"].append(i)
        if len(g["items"]) != len(core):
            raise SystemExit(f"layer {i} shape set differs inside {kind}")
    print("  layer archetypes: " + " · ".join(
        f"{g['kind']} x{len(g['indices'])} ({len(g['items'])} tensors)" for g in groups.values()))

    groups = dict(sorted(groups.items(), key=lambda kv: (kv[1]["kind"], kv[1]["indices"][0])))
    for g in groups.values():
        g["mode"] = "full" if g["kind"] == "full_attention" else "linear"

    io = {n: dict(v, count=1) for n, v in both.items()
          if not LAYER_RE.match(n) and not n.startswith("mtp.") and not n.startswith("visual.")}
    ple = {n: dict(v, count=128 if n.endswith("ngram_embedding.weight") else 1)
           for n, v in both.items()
           if re.match(r"^model\.layers\.\d+\.ple\.", n)}
    mtp = {n: dict(v, count=1) for n, v in both.items() if n.startswith("mtp.")}
    vis = {n: dict(v, count=1) for n, v in both.items() if n.startswith("visual.")}
    for label, d in (("io", io), ("ple", ple), ("mtp", mtp), ("vision", vis)):
        if not d:
            raise SystemExit(f"{label} group is empty — check the fold rules")
    print(f"  io {len(io)} · ple {len(ple)} · mtp {len(mtp)} · vision {len(vis)}")
    return {"groups": groups, "io": io, "ple": ple, "mtp": mtp, "vision": vis}


NOTE = {
    "linear_attn.in_proj_qkv.weight": "Gated DeltaNet: fused Q/K/V projection",
    "linear_attn.in_proj_z.weight": "Gated DeltaNet: output gate",
    "linear_attn.in_proj_a.weight": "Gated DeltaNet: decay (a)",
    "linear_attn.in_proj_b.weight": "Gated DeltaNet: decay (b)",
    "linear_attn.conv1d.weight": "short convolution over the delta-rule state",
    "linear_attn.A_log": "per-head log decay",
    "linear_attn.dt_bias": "delta-rule timestep bias",
    "linear_attn.norm.weight": "output norm",
    "linear_attn.out_proj.weight": "back into the residual stream",
    "self_attn.q_proj.weight": "QSA: 24 query heads at head_dim 256",
    "self_attn.k_proj.weight": "QSA: 2 KV heads",
    "self_attn.v_proj.weight": "QSA: value projection",
    "self_attn.o_proj.weight": "QSA: output projection",
    "self_attn.q_norm.weight": "per-head query norm",
    "self_attn.k_norm.weight": "per-head key norm",
    "self_attn.indexer.index_qk_proj.weight": "indexer: shared Q/K projection (MQA, 4 query heads, 1 key head)",
    "self_attn.indexer.q_layernorm.weight": "indexer query norm",
    "self_attn.indexer.k_layernorm.weight": "indexer key norm",
    "mlp.gate.weight": "router: 512 experts, top-10 (sigmoid)",
    "mlp.experts.gate_up_proj": "routed expert bank: 512 x gate+up",
    "mlp.experts.down_proj": "routed expert bank: 512 x down",
    "mlp.shared_expert.gate_proj.weight": "always-on shared expert: gate",
    "mlp.shared_expert.up_proj.weight": "always-on shared expert: up",
    "mlp.shared_expert.down_proj.weight": "always-on shared expert: down",
    "mlp.shared_expert_gate.weight": "shared-expert output gate",
    "attn_hyper_connection.block_inject_weight.weight": "gated residual: attention branch inject",
    "attn_hyper_connection.input_mix_weight_up.weight": "gated residual: attention branch mix (up)",
    "attn_hyper_connection.input_mix_weight_down.weight": "gated residual: attention branch mix (down)",
    "attn_hyper_connection.hc_norm.weight": "gated residual: attention branch norm",
    "mlp_hyper_connection.block_inject_weight.weight": "gated residual: MoE branch inject",
    "mlp_hyper_connection.input_mix_weight_up.weight": "gated residual: MoE branch mix (up)",
    "mlp_hyper_connection.input_mix_weight_down.weight": "gated residual: MoE branch mix (down)",
    "mlp_hyper_connection.hc_norm.weight": "gated residual: MoE branch norm",
    "embed_tokens.weight": "248,320 padded rows at 2560 dims",
    "lm_head.weight": "248,320 logits, untied",
    "hyper_connection_mixer.input_mix_weight_up.weight": "final gated-residual mixer (up)",
    "hyper_connection_mixer.input_mix_weight_down.weight": "final gated-residual mixer (down)",
    "hyper_connection_mixer.hc_norm.weight": "final gated-residual norm",
    "ple.ple_embedding.ngram_embedding.weight": "20,000,000 n-gram rows in 128 shards (51B params)",
    "ple.ple_embedding.layer_multipliers": "per-layer n-gram scaling",
    "ple.ple_embedding.ngram_heads_offsets": "bigram/trigram head offsets",
    "ple.ple_embedding.ngram_heads_vocab_sizes": "bigram/trigram head sizes",
    "ple.key_proj.weight": "n-gram lookup: key projection",
    "ple.value_proj.weight": "n-gram lookup: value projection",
    "ple.conv1d.weight": "n-gram embedding: short convolution",
    "ple.norm_query.weight": "n-gram embedding: query norm",
    "ple.norm_key.weight": "n-gram embedding: key norm",
    "ple.norm_conv.weight": "n-gram embedding: conv norm",
    "fc_embedding.weight": "MTP: embedding of the next-token stream",
    "fc_hidden.weight": "MTP: hidden-state projection",
    "pre_fc_norm_embedding.weight": "MTP: norm before the embedding projection",
    "pre_fc_norm_hidden.weight": "MTP: norm before the hidden projection",
    "patch_embed.proj.weight": "vision: 16x16 patch projection",
    "pos_embed.weight": "vision: 2304 learned positions",
    "merger.linear_fc1.weight": "vision: spatial-merge projector",
    "merger.linear_fc2.weight": "vision: projector output",
    "merger.norm.weight": "vision: projector norm",
}


def tail_of(name: str) -> str:
    n = re.sub(r"^model\.layers\.\d+\.", "", name)
    n = re.sub(r"^mtp\.layers\.\d+\.", "", n)
    n = re.sub(r"^visual\.blocks\.\d+\.", "", n)
    return re.sub(r"^(model|mtp|visual)\.", "", n)


CAT = [(r"^linear_attn\.", "linear_attn"), (r"^self_attn\.indexer\.", "indexer"),
       (r"^self_attn\.", "qsa"), (r"hyper_connection", "gated"),
       (r"^mlp\.experts\.", "expert"), (r"^mlp\.shared_expert", "shared"),
       (r"^mlp\.gate", "router"), (r"^(ple|.*ngram)", "ngram"),
       (r"^visual\.", "vision"), (r"^mtp\.", "mtp"),
       (r"embed_tokens|lm_head", "vocab"), (r"norm|layernorm|A_log|dt_bias", "norm")]


def cat_of(name: str, group: str = "") -> str:
    n = tail_of(name)
    if group == "ngram":
        return "ngram"
    if group == "vision":
        return "vision"
    if group == "mtp":
        return "mtp" if not re.search(r"^(self_attn|mlp|.*hyper_connection)", n) else cat_of(re.sub(r"^mtp\.", "", name))
    if "embed_tokens" in n or n.startswith("lm_head"):
        return "vocab"
    if n.startswith("hyper_connection_mixer"):
        return "gated"
    for pat, c in CAT:
        if re.search(pat, n):
            return c
    return "other"


def js_array(var: str, items, comment: str, group: str = "") -> str:
    lines = [f"const {var}=[", f"  /* {comment} */"]
    for name, v in items:
        note = NOTE.get(tail_of(name), "")
        extra = f",{v['count']}" if v.get("count", 1) > 1 else ""
        label = name + (f"  ({note})" if note else "")
        lines.append(f'  W({json.dumps(label)},{json.dumps(v["dims"])},{json.dumps(cat_of(name, group))},'
                     f'"",{{bf16:{v["b16"]},fp8:{v["b8"]},nvfp4:{v["b4"]}}}{extra}),')
    lines.append("];")
    return "\n".join(lines)


DATA_HEAD = r"""// ---------- data.js ----------
/* Qwen3.8-Flash-Next — Tensor Atlas. Architecture from the published config of
   Qwen/Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration, 48 layers as 12 x (3 x (Gated DeltaNet
   -> MoE) -> 1 x (Qwen Sparse Attention -> MoE))); every byte measured from the safetensors headers
   of the three checkpoints named in Sources, summed per logical tensor with quantisation scales and
   packed weights folded into the weight they describe. */
const CFG = @@CFG@@;
const CT=CFG, VC=@@VISIONCFG@@;
const CFG_FULL=@@CFGFULL@@;   /* the published wrapper config, shown in the config panel */
"""

DATA_TAIL = r"""const COL={blue:'#638bff',enc:'#5ca7ff',dec:'#55d7c1',auxa:'#b99bff',linear_attn:'#55ddd0',qsa:'#c79bff',indexer:'#9aa8ff',gated:'#ffb248',expert:'#66deb0',shared:'#d6e99c',router:'#f4b765',ngram:'#4fc8e8',vision:'#62d5d0',mtp:'#e9a6dd',vocab:'#c4d0ff',norm:'#a7b5cc',muted:'#8390a6',head:'#c4d0ff',full:'#efbc71',lin:'#9bb6d8',attn:'#55ddd0',expertA:'#66deb0',sharedA:'#d6e99c',mtpA:'#e9a6dd',visionA:'#62d5d0',indexA:'#9aa8ff',ar_attn:'#55ddd0',nar_attn:'#c79bff',engram:'#b99bff',mhc:'#c9a3fa',vae:'#4fc8e8',latent:'#e9a6dd',time:'#e9a6dd'};
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const lerp=(a,b,t)=>a+(b-a)*t;
const smooth=x=>x*x*(3-2*x);
const num=x=>Math.round(x).toLocaleString('en-US');
function fmtP(p){return p>=1e12?(p/1e12).toFixed(3)+'T':p>=1e9?(p/1e9).toFixed(2)+'B':p>=1e6?(p/1e6).toFixed(2)+'M':p>=1e3?(p/1e3).toFixed(1)+'K':num(p);}
function bytes(n,binary=false){let b=binary?1024:1000,u=binary?['B','KiB','MiB','GiB','TiB']:['B','KB','MB','GB','TB'],k=0;while(n>=b&&k<4){n/=b;k++;}return (k===0?num(n):n.toFixed(n>=100?1:2))+' '+u[k];}
const SOURCE_BASE='https://huggingface.co/Qwen/Qwen3.8-Flash-Next';
const FP8_MODEL='https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8';
const NVFP4_MODEL='https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4';
const SOURCES=[
 ['Model card',SOURCE_BASE,'Qwen3.8-Flash-Next: an experimental preview of the architecture that will underpin Qwen4. 125B parameters with 6B activated, plus a 51B n-gram embedding and a 4B MTP layer. Hybrid attention pairs Gated DeltaNet with Qwen Sparse Attention, Gated Residual widens the residual stream, and the MoE routes to 512 experts.'],
 ['config.json',SOURCE_BASE+'/blob/main/config.json','Embedded field for field: 48 layers as 12 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Qwen Sparse Attention -> MoE)), hidden 2560, head_dim 256 with a 64-dim RoPE slice, 24 query heads and 2 KV heads, 512 experts with top-10 plus one shared expert at intermediate 640, indexer budget 2048 tokens (512 blocks), n-gram embedding 20,000,000 at layer 2, 262,144 native context.'],
 ['BF16 checkpoint',SOURCE_BASE+'/tree/main','131 shards, 1658 tensors, 360.0002 GB: the only published full-precision copy, and the reference every other mode is measured against.'],
 ['FP8 checkpoint',FP8_MODEL+'/tree/main','Qwen\u2019s own compressed-tensors build: e4m3 with 128x128 weight blocks, 185.5233 GB over 131 shards. 943 module patterns stay unquantised (norms, routers, gates, most hyper-connection tensors, the vision tower), while the n-gram embedding is explicitly converted.'],
 ['NVFP4 checkpoint',NVFP4_MODEL+'/tree/main','NVIDIA\u2019s modelopt build: 4-bit weights stored as weight_packed with per-block scale and a global scale, 132.6802 GB over 11 shards.'],
 ['Engine', 'https://github.com/nisten/LLMViz-DeepSeek-V4.1-Flash','The Atlas v4 engine (MIT, \u00a9 2026 netsin) is reused whole; only the model half is replaced.'],
 ['Method',SOURCE_BASE,'Every figure on this page is the sum of tensor byte ranges read from those shard headers over HTTP Range. No weights were downloaded, and no number here is a shape x bytes estimate.']
];
const MODE_INFO={
 bf16:{label:'BF16',short:'BF16',color:COL.enc,note:'Qwen/Qwen3.8-Flash-Next · 131 shards · 2 bytes per parameter'},
 fp8:{label:'FP8',short:'FP8',color:COL.dec,note:'Qwen/Qwen3.8-Flash-Next-FP8 · 131 shards · e4m3, 128x128 weight blocks'},
 nvfp4:{label:'NVFP4',short:'NVFP4',color:COL.indexer,note:'nvidia/Qwen3.8-Flash-Next-NVFP4 · 11 shards · 4-bit packed + global scales'}
};
const MODE_KEYS=['bf16','fp8','nvfp4'];
function modeFor(i){return LAYER_TYPES[i]==='full_attention'?'full':'lin';}
function ownerFor(i){return i;}
function indexOwnerFor(i){return LAYER_TYPES[i]==='full_attention'?i:null;}
function W(name,shape,cat,note='',ex=null,count=1){
 return {name,shape,cat,note,count,
   p:shape.reduce((a,b)=>a*b,1)*count,
   format:ex&&ex.nvfp4&&ex.nvfp4<ex.bf16*0.4?'nvfp4':(ex&&ex.fp8&&ex.fp8<ex.bf16*0.9?'fp8':'bf16'),
   ex};
}
function wBytes(w,mode='bf16'){return w.ex?w.ex[mode]:2*w.p;}
function wFormat(w,mode){return {bf16:'BF16',fp8:'FP8 / 128x128 block scales',nvfp4:'NVFP4 / packed + global scale'}[mode];}
const sumP=ws=>ws.reduce((a,w)=>a+w.p,0);
const sumB=(ws,m)=>ws.reduce((a,w)=>a+wBytes(w,m),0);
@@TABLES@@
const FP8_CONFIG=@@FP8CFG@@;
const NV_CONFIG=@@NVCFG@@;
const LAYER_TYPES=@@LAYERTYPES@@;
function layerWeights(i){return LAYER_TYPES[i]==='full_attention'?LAYER_FULL_W:LAYER_LIN_W;}
const NLAYERS=48;
const LAYERS=Array.from({length:48},(_,i)=>({id:'L'+i,index:i,label:'Layer '+String(i).padStart(2,'0'),
 part:LAYER_TYPES[i]==='full_attention'?'decoder':'encoder',mode:modeFor(i),owner:i,
 indexOwner:indexOwnerFor(i),ratio:1,ws:layerWeights(i)}));
const AUX_TABLES=[];
const ENGRAM=[];
const EMBED={id:'embed',label:'Token embedding',ws:EMBED_W};
const HEAD={id:'head',label:'Untied output head',ws:HEAD_W};
const MIXER={id:'mixer',index:48,label:'Gated-residual mixer (final)',ws:MIXER_W};
const NGRAM={id:'ngram',index:49,label:'N-gram embedding (51B params)',ws:NGRAM_W};
const MTP={id:'mtp',index:50,label:'MTP layer (4B params)',ws:MTP_W};
const VISION={id:'vision',index:51,label:'Vision tower (27 blocks)',ws:VISION_W};
const MODULES=[EMBED,...LAYERS,HEAD,MIXER,NGRAM,MTP,VISION];
const ALL_W=MODULES.flatMap(m=>m.ws);
const TOTALS=Object.fromEntries(MODE_KEYS.map(m=>[m,sumB(ALL_W,m)]));
const TOTAL_P=sumP(ALL_W);
const FP8_DELTA=TOTALS.bf16-TOTALS.fp8;
const NV_DELTA=TOTALS.bf16-TOTALS.nvfp4;
const MOE_EXPERT_P=3*2560*640;   /* gate + up + down per routed expert */
const KV_FULL_PER_TOKEN=12*2*2*256*2;
const KV_LINEAR_STATE=18*16*128*4*4;
const CATEGORIES=[
 ['linear_attn','Gated DeltaNet (36 layers)',COL.linear_attn,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='linear_attn'))],
 ['qsa','Qwen Sparse Attention (12 layers)',COL.qsa,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='qsa'))],
 ['indexer','QSA indexer',COL.indexer,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='indexer'))],
 ['expert','Routed experts (512 x 48)',COL.expert,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='expert'))],
 ['shared','Shared expert',COL.shared,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='shared'))],
 ['router','Routers and gates',COL.router,LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='router'))],
 ['gated','Gated Residual mixers',COL.gated,[...LAYERS.flatMap(l=>l.ws.filter(w=>w.cat==='gated')),...MIXER.ws]],
 ['ngram','N-gram embedding',COL.ngram,NGRAM.ws],
 ['mtp','MTP layer',COL.mtp,MTP.ws],
 ['vision','Vision tower',COL.vision,VISION.ws],
 ['vocab','Embedding + head',COL.vocab,[...EMBED.ws,...HEAD.ws]],
 ['norm','Norms',COL.norm,ALL_W.filter(w=>w.cat==='norm')]
];
const EXP={
 overview:{title:'Twelve rounds of three plus one.',body:'Every layer of Qwen3.8-Flash-Next runs either Gated DeltaNet (a linear-attention layer with a constant-size state) or Qwen Sparse Attention (a full-attention layer that keeps a KV cache). The stack repeats the pattern 3 : 1 twelve times, so 36 layers are DeltaNet and 12 are QSA. Both feed a 512-expert MoE, and every layer is wrapped in Gated Residual mixers that widen the residual stream.'},
 full:{title:'QSA: full attention with a sparse budget.',body:'12 of 48 layers keep a KV cache: 2 KV heads at head_dim 256. A MQA indexer (4 query heads, 1 shared key head, head_dim 128) scores micro-blocks and spends a budget of 512 blocks (2048 tokens) per query, so the layer reads a slice of the context instead of all of it.'},
 linear:{title:'Gated DeltaNet: constant state, not a cache.',body:'36 of 48 layers carry a linear-attention state: 48 value heads and 16 query/key heads at head_dim 128, with a short convolution over the delta-rule state, learned per-head decays and a gating projection. Their memory does not grow with the context, which is what makes 262,144 tokens affordable here.'},
 bf16:{title:'The reference: 360 GB, all BF16.',body:'131 shards, 1658 tensors. Two thirds of it is routed experts and the n-gram table; the rest is the hybrid attention stacks, the gated-residual mixers, the MTP layer and the vision tower.'},
 fp8:{title:'Qwen\u2019s own FP8 build.',body:'185.5233 GB in 131 shards: e4m3 weights with 128x128 blocks, and 943 module patterns deliberately left unquantised (norms, routers, gates, most hyper-connection tensors, the vision tower). The n-gram embedding is converted.'},
 nvfp4:{title:'NVIDIA\u2019s NVFP4 build is the smallest.',body:'132.6802 GB over 11 shards: 4-bit weights with a per-block scale and a global scale each. Fewer shards, more tensors, and the smallest payload of the three.'},
 storage:{title:'Where 360 GB goes.',body:'Routed experts are the mass: 48 layers x two banks of 512 experts. Then the n-gram embedding (51B parameters), the hybrid attention stacks, the gated-residual mixers, the MTP layer and the vision tower.'},
 cache:{title:'Only 12 layers keep a cache.',body:'The QSA layers cache 2 KV heads x 256 dims x (k+v) x 2 bytes = 4,096 bytes per token per layer: 49,152 bytes per token across the 12 of them. The other 36 layers hold a constant delta-rule state, so the growing part of the memory is a quarter of the stack - and the indexer adds 512-block sparse reads on top.'}
};
const BENCH=@@BENCH@@;
const BENCH_MODELS=@@BENCHMODELS@@;
function pickExperts(seed,n=512,k=10){return Array.from({length:k},(_,j)=>(seed*7+j*13)%n);}
const PHASES=[
 {name:'Embed',label:'Embed',from:0,to:4,active:'248,320 + 20M',color:COL.vocab,caption:'Token ids, then n-grams.',desc:'The token embedding is 248,320 padded rows. Layer 2 adds the n-gram embedding: 20,000,000 bigram/trigram rows looked up by hash - 51B parameters that scale the model without scaling the compute.'},
 {name:'DeltaNet',label:'DeltaNet',from:4,to:9,active:'36 layers',color:COL.linear_attn,caption:'Constant-state linear attention.',desc:'Gated DeltaNet layers carry 48 value heads and 16 query/key heads at head_dim 128, a short convolution over the delta-rule state, learned decays and an output gate. Their state does not grow with the context.'},
 {name:'MoE',label:'MoE',from:9,to:14,active:'512 experts',color:COL.expert,caption:'Top-10 of 512, plus one shared.',desc:'Every layer routes to 10 of 512 experts at intermediate 640 and always runs one shared expert: the two banks hold most of the checkpoint.'},
 {name:'QSA',label:'QSA',from:14,to:19,active:'12 layers',color:COL.qsa,caption:'Sparse micro-block attention.',desc:'Every fourth layer is full attention: 24 query heads, 2 KV heads at head_dim 256, and an indexer that spends a 512-block budget per query instead of reading the whole context.'},
 {name:'Decode',label:'Decode',from:19,to:22,active:'+ MTP',color:COL.mtp,caption:'Multi-token prediction and the head.',desc:'One MTP layer drafts the next tokens from the hidden state; the untied head projects to 248,320 logits.'}
];
const TRAIN_PHASES=[
 {name:'Pretrain',from:0,to:8,active:'Muon + AdamW',color:COL.expert,caption:'Two optimizers, two weight groups.',desc:'Schematic: the card describes Muon and AdamW applied to specific weight categories, with no batch-size warmup.'},
 {name:'Post-train',from:8,to:14,active:'agentic',color:COL.qsa,caption:'Agentic and multimodal work.',desc:'Schematic: post-training covers the agentic and vision-language behaviour this card reports benchmarks for.'},
 {name:'Quantise',from:14,to:20,active:'FP8 / NVFP4',color:COL.gated,caption:'What the two published builds convert.',desc:'Schematic: FP8 keeps 943 module patterns unquantised; NVFP4 packs weights to 4 bits with two scales each.'}
];
function phasesFor(mode){return mode==='training'?TRAIN_PHASES:PHASES;}
function phaseAt(t,mode='inference'){const ps=phasesFor(mode);return ps.find(p=>t>=p.from&&t<p.to)||ps[ps.length-1];}
const TC={ar_attn:'#55ddd0',nar_attn:'#c79bff',linear_attn:'#55ddd0',qsa:'#c79bff',indexer:'#9aa8ff',gated:'#ffb248',expert:'#66deb0',shared:'#d6e99c',router:'#f4b765',ngram:'#4fc8e8',vision:'#62d5d0',mtp:'#e9a6dd',vocab:'#c4d0ff',norm:'#a7b5cc',head:'#c4d0ff',attn:'#55ddd0',kv:'#61dccb',local:'#ffc477',out:'#b2bfff',q:'#8caaff',engram:'#b99bff',mhc:'#c9a3fa'};
function tensorKind(w){return TC[w.cat]?w.cat:'head';}
function tensorColor(w){return TC[tensorKind(w)]||COL.linear_attn;}
function tensorShort(w){return w.name.replace(/^model\.layers\.\d+\./,'').replace(/^(model|mtp|visual)\./,'').replace(/\.weight$/,'');}
function weightParts(w,mode){const total=wBytes(w,mode);return {data:total,scales:0,aux:0,total};}
function displayWeights(m){return m.ws;}
function findWeight(m,name){return displayWeights(m).find(w=>w.name===name)||m.ws.find(w=>w.name===name);}
function orderWeights(m){const ws=displayWeights(m);const rank=w=>{const n=w.name;
 if(n.includes('linear_attn.in_proj_qkv'))return 0; if(n.includes('linear_attn.in_proj_z'))return 1;
 if(n.includes('linear_attn.in_proj_a'))return 2; if(n.includes('linear_attn.in_proj_b'))return 3;
 if(n.includes('linear_attn.conv1d'))return 4; if(n.includes('linear_attn.A_log')||n.includes('linear_attn.dt_bias'))return 5;
 if(n.includes('linear_attn.norm'))return 6; if(n.includes('linear_attn.out_proj'))return 7;
 if(n.includes('self_attn.q_proj'))return 0; if(n.includes('self_attn.k_proj'))return 1; if(n.includes('self_attn.v_proj'))return 2;
 if(n.includes('self_attn.o_proj'))return 3; if(n.includes('self_attn.q_norm')||n.includes('self_attn.k_norm'))return 4;
 if(n.includes('indexer.index_qk_proj'))return 5; if(n.includes('indexer.q_layernorm')||n.includes('indexer.k_layernorm'))return 6;
 if(n.includes('attn_hyper_connection'))return 8; if(n.includes('mlp_hyper_connection'))return 12;
 if(n.includes('mlp.experts.gate_up_proj'))return 9; if(n.includes('mlp.experts.down_proj'))return 10;
 if(n.includes('mlp.gate.weight'))return 11; if(n.includes('shared_expert_gate'))return 13; if(n.includes('shared_expert'))return 14;
 if(n.includes('embed_tokens'))return 0; if(n.includes('lm_head'))return 1;
 if(n.includes('ngram_embedding'))return 0; if(n.includes('ple.'))return 1;
 return 20;};return [...ws].sort((a,b)=>rank(a)-rank(b));}
function isAttentionTensor(w){return w.cat==='qsa'||w.cat==='indexer';}
function tensorInfo(w,m){
 const p=fmtP(w.p),size=bytes(wBytes(w,'bf16'));
 let t='Stored tensor.',b='Two bytes per parameter in the BF16 reference. Switch the precision to see what the FP8 and NVFP4 builds store for it.';
 const n=w.name;
 if(n.includes('linear_attn.in_proj_qkv')){t='Gated DeltaNet: fused Q/K/V projection.';b='16 query/key heads and 48 value heads at head_dim 128 are projected together, then the delta-rule recurrence updates a state whose size does not depend on the context.';}
 else if(n.includes('linear_attn.in_proj_z')){t='Gated DeltaNet: output gate.';b='The gating projection that decides how much of the delta-rule output reaches the residual stream.';}
 else if(n.includes('linear_attn.in_proj_a')||n.includes('linear_attn.in_proj_b')){t='Gated DeltaNet: decay parameters.';b='Two projections feed the per-head decay (a) and its bias (b): the forget gate of the linear attention.';}
 else if(n.includes('linear_attn.conv1d')){t='Gated DeltaNet: short convolution.';b='A 4-tap convolution over the delta-rule state, applied before the recurrence.';}
 else if(n.includes('linear_attn.A_log')){t='Gated DeltaNet: per-head log decay.';b='The learned, per-head decay used to keep the recurrent state stable.';}
 else if(n.includes('linear_attn.dt_bias')){t='Gated DeltaNet: timestep bias.';b='Bias on the delta-rule step size.';}
 else if(n.includes('linear_attn.out_proj')){t='Gated DeltaNet: output projection.';b='Brings the 48-head delta-rule output back to the 2560-wide residual stream.';}
 else if(n.includes('linear_attn.norm')){t='Gated DeltaNet: output norm.';b='Normalises the recurrent output before the projection.';}
 else if(n.includes('indexer.index_qk_proj')){t='QSA indexer: shared Q/K projection.';b='Multi-query attention with 4 query heads and 1 shared key head at head_dim 128: it scores micro-blocks so the layer can spend a 512-block budget instead of reading every token.';}
 else if(n.includes('indexer.')){t='QSA indexer: layer norm.';b='Normalises the indexer queries or keys before scoring.';}
 else if(n.includes('self_attn.q_proj')){t='QSA: query projection.';b='24 query heads at head_dim 256, of which 64 dims carry RoPE.';}
 else if(n.includes('self_attn.k_proj')||n.includes('self_attn.v_proj')){t='QSA: key/value projection.';b='2 KV heads at head_dim 256 - the only layers in this model that grow a cache with the context.';}
 else if(n.includes('self_attn.o_proj')){t='QSA: output projection.';b='Back into the widened residual stream.';}
 else if(n.includes('self_attn.q_norm')||n.includes('self_attn.k_norm')){t='QSA: per-head norm.';b='128 scales applied per head before the attention dot product.';}
 else if(n.includes('mlp.experts.gate_up_proj')){t='Routed experts: gate and up bank.';b='One tensor holds all 512 experts\u2019 gate and up projections at intermediate 640. Top-10 are activated per token, sigmoid routing.';}
 else if(n.includes('mlp.experts.down_proj')){t='Routed experts: down bank.';b='The matching 512 down projections. Experts are the largest single mass in the checkpoint.';}
 else if(n.includes('mlp.gate.weight')){t='Router.';b='Scores 512 experts per token - kept at higher precision in both quantised builds.';}
 else if(n.includes('shared_expert_gate')){t='Shared-expert gate.';b='Scales the always-on expert\u2019s contribution.';}
 else if(n.includes('shared_expert')){t='Shared expert.';b='One expert that runs for every token, alongside the routed ones.';}
 else if(n.includes('attn_hyper_connection')){t='Gated Residual: attention branch.';b='Four tensors per layer widen, mix and re-inject the residual stream around the attention block: an element-wise read gate plus a scalar write gate.';}
 else if(n.includes('mlp_hyper_connection')){t='Gated Residual: MoE branch.';b='The same mechanism around the MoE block, with the branch injection weight.';}
 else if(n.includes('hyper_connection_mixer')){t='Final gated-residual mixer.';b='The last Gated Residual stage, after all 48 layers.';}
 else if(n.includes('ngram_embedding')){t='N-gram embedding table.';b='20,000,000 bigram/trigram rows in 128 shards: 51B parameters that add capacity without adding per-token compute, which is why the card calls them easier to offload than experts.';}
 else if(n.includes('ple.')){t='N-gram lookup plumbing.';b='Key/value projections, a short convolution and three norms that turn hashed n-grams into an embedding the layer can add.';}
 else if(n.includes('fc_embedding')||n.includes('fc_hidden')){t='MTP projection.';b='The MTP layer fuses the next-token embedding with the hidden state (4B parameters) to draft more than one token per step.';}
 else if(n.includes('pre_fc_norm')){t='MTP norm.';b='Normalises the embedding or hidden side before the MTP fusion.';}
 else if(n.includes('patch_embed')){t='Vision: patch embedding.';b='16x16 pixels per patch with a temporal patch of 2 frames.';}
 else if(n.includes('pos_embed')){t='Vision: position embedding.';b='2,304 learned positions for the vision tower.';}
 else if(n.includes('visual.blocks')){t='Vision block.';b='One of 27 blocks in the vision encoder (1152-wide, 16 heads) that feeds the merger.';}
 else if(n.includes('merger')){t='Vision: spatial merger.';b='Merges 2x2 patch groups and projects them to the 2560 dims the language model reads.';}
 else if(n.includes('embed_tokens')){t='Token embedding.';b='248,320 padded rows at 2560 dims, untied from the head.';}
 else if(n.includes('lm_head')){t='Output head.';b='2560 to 248,320 logits in its own table: untied, and left unquantised by the FP8 build.';}
 else if(w.cat==='norm'){t='Norm.';b='RMSNorm (eps 1e-6). Norms stay high precision in both quantised builds.';}
 return {title:t,body:b,size,p};}
function layerStory(m){
 if(!m.mode)return null;
 const del=m.mode==='lin';
 return {title:del?'Gated DeltaNet layer':'Qwen Sparse Attention layer',
  body:del?'A linear-attention layer: fused QKV projection, a 4-tap convolution, learned per-head decays and an output gate, followed by a 512-expert MoE. Its state is constant-size, so it costs the same at 8K and at 262K tokens.'
       :'A full-attention layer: 24 query heads and 2 KV heads at head_dim 256, plus an indexer that scores micro-blocks and reads a 512-block budget. Every fourth layer of the stack is one of these twelve.',
  detail:'24 tensors per layer: the attention block, four Gated Residual tensors around it, the router, the two expert banks, the shared expert and its gate, plus four more Gated Residual tensors around the MoE.'};}
"""

REGEX_SUBS: list[tuple[str, str]] = []


def build_data_js(t: dict) -> str:
    groups, io, ple, mtp, vis = t["groups"], t["io"], t["ple"], t["mtp"], t["vision"]
    lin = next(g for g in groups.values() if g["mode"] == "linear")
    full = next(g for g in groups.values() if g["mode"] == "full")
    embed = sorted((n, v) for n, v in io.items() if "embed_tokens" in n)
    head = sorted((n, v) for n, v in io.items() if n.startswith("lm_head"))
    mixer = sorted((n, v) for n, v in io.items() if "hyper_connection_mixer" in n)
    placed = {n for g in (embed, head, mixer) for n, _ in g}
    missing = sorted(set(io) - placed)
    if missing:
        raise SystemExit(f"top-level tensors placed in no module: {missing[:6]}")

    tables = "\n".join([
        js_array("LAYER_LIN_W", sorted(lin["items"].items()),
                 "the 36 Gated DeltaNet (linear attention) layers", "layer"),
        js_array("LAYER_FULL_W", sorted(full["items"].items()),
                 "the 12 Qwen Sparse Attention layers", "layer"),
        js_array("EMBED_W", embed, "the token embedding"),
        js_array("HEAD_W", head, "the untied output head"),
        js_array("MIXER_W", mixer, "the final gated-residual mixer", "io"),
        js_array("NGRAM_W", sorted(ple.items()), "the 20M-entry n-gram embedding and its lookup plumbing", "ngram"),
        js_array("MTP_W", sorted(mtp.items()), "the multi-token-prediction layer", "mtp"),
        js_array("VISION_W", sorted(vis.items()), "the 27-block vision tower", "vision"),
    ])
    wrapper = dict(load(CFG_BASE))
    tc = dict(wrapper["text_config"])
    tc["architectures"] = wrapper.get("architectures")
    tc["model_type"] = tc.get("model_type")
    tc["vision_config"] = wrapper.get("vision_config", {})
    tc["layer_types"] = layer_types()
    tc["kv_source_layer_ids"] = [i for i, k in enumerate(layer_types()) if k == "full_attention"]
    tc["index_source_layer_ids"] = list(tc["kv_source_layer_ids"])
    cfg = tc
    js = DATA_HEAD + DATA_TAIL
    js = js.replace("@@TABLES@@", tables)
    js = js.replace("@@CFG@@", json.dumps(cfg, ensure_ascii=False))
    js = js.replace("@@CFGFULL@@", json.dumps(wrapper, ensure_ascii=False))
    js = js.replace("@@VISIONCFG@@", json.dumps(cfg.get("vision_config", {}), ensure_ascii=False))
    js = js.replace("@@FP8CFG@@", json.dumps(load(CFG_FP8), ensure_ascii=False))
    js = js.replace("@@NVCFG@@", json.dumps(load(CFG_NV), ensure_ascii=False))
    js = js.replace("@@LAYERTYPES@@", json.dumps(layer_types()))
    js = js.replace("@@BENCH@@", json.dumps(load(DIR / "bench.json"), ensure_ascii=False))
    js = js.replace("@@BENCHMODELS@@", json.dumps(load(DIR / "bench-models.json"), ensure_ascii=False))
    return js


def main() -> int:
    import objectcode
    import panels
    t = templates()
    measured = {}
    for mode, p in (("bf16", BF_JSON), ("fp8", FP8_JSON), ("nvfp4", NV_JSON)):
        if p.exists():
            measured[mode] = load(p)["payload_bytes"]
        else:
            print(f"!! {p.name} missing — measure it before building (mode {mode} unchecked)")
    groups, io, ple, mtp, vis = t["groups"], t["io"], t["ple"], t["mtp"], t["vision"]
    lin = next(g for g in groups.values() if g["mode"] == "linear")
    full = next(g for g in groups.values() if g["mode"] == "full")
    key = {"bf16": "b16", "fp8": "b8", "nvfp4": "b4"}

    def page_total(mode: str) -> int:
        k = key[mode]
        return (sum(v[k] for v in lin["items"].values()) * len(lin["indices"])
                + sum(v[k] for v in full["items"].values()) * len(full["indices"])
                + sum(v[k] for v in io.values()) + sum(v[k] for v in ple.values())
                + sum(v[k] for v in mtp.values()) + sum(v[k] for v in vis.values()))

    print(f"data · {len(lin['indices'])} linear + {len(full['indices'])} full layers "
          f"({len(lin['items'])}/{len(full['items'])} tensors) + {len(io)} io + {len(ple)} ngram "
          f"+ {len(mtp)} mtp + {len(vis)} vision")
    for m, want in measured.items():
        got = page_total(m)
        if got != want:
            print(f"GATE FAIL page {m} {got:,} vs measured {want:,} (delta {got-want:,})")
            return 3
    print("GATE payload == measured ✔ " + " · ".join(f"{m} {page_total(m)/1e9:.4f} GB" for m in measured))

    if "--data-only" in sys.argv:
        (DIR / "data.js").write_text(build_data_js(t), encoding="utf-8")
        print("wrote data.js")
        return 0

    data_js = build_data_js(t)
    lines = SRC.read_text(encoding="utf-8").split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.startswith("const CFG = {"))
    i1 = next(i for i, l in enumerate(lines) if l.startswith("class CanvasRenderer{"))
    text = "\n".join(lines[:i0] + data_js.split("\n") + [""] + lines[i1:])
    print(f"splice · lines {i0+1}..{i1} -> data.js ({len(data_js.splitlines())} lines)")

    text, rep = objectcode.apply_rewrites(text, panels.REWRITES)
    print(f"panels · {len(rep)} whole-line rewrites applied (structure checked)")

    n_re = 0
    for pat, sub in list(REGEX_SUBS) + list(panels.REGEX_EXTRA):
        text, k = re.subn(pat, lambda _m, sub=sub: sub, text)
        n_re += k
    lit = [(a, b) for a, b in panels.LITERAL_SUBS if a in text]
    print(f"code   · {n_re} regex substitutions, {len(lit)} literal subs "
          f"({len(panels.LITERAL_SUBS) - len(lit)} superseded)")
    text, rep2 = objectcode.apply_literals(text, lit)

    must_have = ["Qwen3.8-Flash-Next", "Gated DeltaNet", "Qwen Sparse Attention", "512", "NVFP4",
                 "n-gram", "MTP", "262,144"]
    must_not = ["DeepSeek V4", "DSpark", "Engram table", "CSA2", "890", "552B", "wo_a", "GLM-5.3",
                "YuE2", "nar_self_attn"]
    body = text.split("</head>", 1)[-1]
    scan = body.replace("deepseek_sparse_attention", "<layer-type>")
    miss = [tok for tok in must_have if tok.lower() not in scan.lower()]
    bad = [tok for tok in must_not if tok.lower() in scan.lower()]
    if miss:
        print(f"RESIDUE FAIL missing: {miss}")
        return 5
    if bad:
        for tok in bad:
            k = scan.lower().find(tok.lower())
            print(f"RESIDUE {tok!r} at {k}: ...{scan[max(0,k-110):k+90]!r}...")
        return 6

    import subprocess
    import tempfile
    pre = DIR / "preflight.py"
    if pre.exists():
        st = subprocess.run([sys.executable, str(pre), "--selftest"], capture_output=True, text=True)
        print((st.stdout or st.stderr).strip().splitlines()[-1])
        if st.returncode != 0:
            print("PREFLIGHT SELF-TEST FAILED — the guard is broken, refusing to ship", file=sys.stderr)
            return 8
        with tempfile.TemporaryDirectory() as td:
            cand = Path(td) / "candidate"
            cand.mkdir()
            (cand / "index.html").write_text(text, encoding="utf-8")
            r = subprocess.run([sys.executable, str(pre), str(cand)], capture_output=True, text=True)
            print(r.stdout.strip())
            if r.returncode != 0:
                print("PREFLIGHT FAILED — index.html NOT written", file=sys.stderr)
                return 7
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.name} · {len(text):,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
