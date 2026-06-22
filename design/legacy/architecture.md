# ViT Image Tokenizer — Legacy Spec (reference)

From-scratch spec of the seamstress ViT image tokenizer that worked well. Reference for the
shoot-out, not a quickdraw component. It maps a clip `(B,T,3,H,W)` to one embedding per frame
`(B,T,d)`; time is folded into the batch, so it is an ordinary per-frame image ViT.

**Provenance.** Run started **2026-03-17 17:57 UTC**, commit `6c117a9` (twin of linked `d2c1231`),
2× H100, `bf16-mixed`, seed 39, `learning=learning_figs diffusion.enabled=false`. Values below are
from that run's logged config.

Source: `learning/models/tokenizers/image_tokenizer_vit.py`,
`learning/models/transformers/{blocks,attention,positional,utils}.py`.

## Trained config

| Param | Value | | Param | Value |
|---|---|---|---|---|
| `image_hw` | `(72,128)` | | `norm_first` | `False` (post-norm) |
| `in_chans` | 3 (ImageNet-norm) | | norm | LayerNorm |
| `dim_latent` | 128 | | FFN | GELU |
| `patch_size` | 8 → 144 patches +CLS = **145 tokens** | | pos-emb | learned absolute |
| `depth` | 2 | | pooling | CLS |
| `heads` | 4 → head_dim 32 | | RoPE/GQA/SwiGLU/QK-norm/logit-cap | off |
| `mlp_ratio` | 4.0 (hidden 512) | | dropout | 0 |

Training: AdamW `lr=1e-3 wd=1e-4`, `grad_clip=1.0`, `batch=172`, `max_epochs=1024`, `bf16-mixed`,
clips of `T=32` frames.

## Shape trace

```
(B,T,3,72,128)
 reshape            → (B·T,3,72,128)
 Conv2d(3→128,k=8,s=8) → (B·T,128,9,16)
 flatten+transpose  → (B·T,144,128)
 prepend CLS        → (B·T,145,128)
 + learned pos_embed→ (B·T,145,128)
 × depth blocks (full bidirectional self-attn, non-causal)
 LayerNorm; take CLS (token 0) → (B·T,128)
 reshape            → (B,T,128)
```

## Components

- **Patch embed:** `nn.Conv2d(in_chans, dim, k=patch, s=patch)`. Use `reshape` (input may be
  non-contiguous). Require `H,W % patch == 0`.
- **CLS token + pos_embed:** `nn.Parameter`, `trunc_normal_(std=0.02)`.
- **Block (post-norm):** `x = norm1(x + attn(x)); x = norm2(x + mlp(x))`. Non-causal, no mask.
- **Attention:** Q/K/V/out linears; reshape to `(B,T,heads,head_dim)`; `scale = head_dim**-0.5`;
  softmax; out-proj.
- **FFN:** `Linear(d→512) → GELU → Linear(512→d)`.
- **Output:** `LayerNorm`, then CLS token as the per-frame embedding.

## Attention acceleration (recommendation)

Drop the seamstress `flash_attn` dependency. The trained config is full bidirectional attention with
no mask and no logit cap, so use **`torch.nn.functional.scaled_dot_product_attention`** — flash-backed,
no extra deps, no compile step. If a custom score/mask is ever enabled (soft-cap, sliding window,
GQA sparsity), switch that path to **FlexAttention** (`torch.nn.attention.flex_attention` under
`torch.compile`) — it keeps the fused kernel where SDPA cannot. Both ship in PyTorch.

## Available knobs (all off in the trained run)

`norm_first` (pre-norm), `use_rmsnorm`, `use_swiglu`, `use_rope` (`theta=10000`, interleaved),
`use_qk_norm`, `use_gqa` + `num_kv_heads`, `attn_logit_cap` (tanh soft-cap → FlexAttention). The
trained model used none of them.
