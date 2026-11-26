"""Read-only alarm for local, dense Llama/Qwen2/Qwen3 causal LM checkpoints."""
import argparse
import json
import math
from pathlib import Path

import torch

from scoring import score

CONTEXTS = (
    "The river ran quietly past the old mill on a grey morning.",
    "She packed two sandwiches, an apple and a map for the hike.",
    "Prices at the market rose slightly after the long dry summer.",
    "A good recipe lists every ingredient before the first step.",
    "The committee will meet again next Thursday to review the budget.",
    "Most cats sleep for more than twelve hours a day.",
    "He repaired the bicycle chain with a small pair of pliers.",
    "The museum opens at nine and closes at five on weekdays.",
)


def lens(mlp, gain, head):
    sign = (mlp.gate_proj.weight * mlp.up_proj.weight).sum(1).sign()
    sign[sign == 0] = 1
    directions = (mlp.down_proj.weight.T * sign[:, None]) * gain
    directions /= directions.norm(dim=1, keepdim=True).clamp_min(1e-30)
    rows = head - head.mean(0)
    rows /= rows.norm(dim=1, keepdim=True).clamp_min(1e-30)
    best = directions.new_full((len(directions),), -2)
    ids = torch.zeros_like(best, dtype=torch.long)
    for offset in range(0, len(rows), 4096):
        val, idx = (directions @ rows[offset:offset + 4096].T).max(1)
        update = val > best
        best[update], ids[update] = val[update], idx[update] + offset
    return bool((best >= .8).sum() >= 2), sorted(set(ids.tolist()))


def states(model, seqs):
    mlp = model.model.layers[-1].mlp
    device = mlp.down_proj.weight.device
    width = max(map(len, seqs))
    x = torch.tensor([s + [0] * (width - len(s)) for s in seqs], device=device)
    idx = (torch.arange(len(seqs), device=device), torch.tensor([len(s) - 1 for s in seqs], device=device))
    out = {}
    hooks = []
    try:
        hooks.append(mlp.register_forward_hook(lambda _, a, o: out.update(m=o[idx])))
        hooks.append(mlp.down_proj.register_forward_pre_hook(lambda _, a: out.update(a=a[0][idx])))
        hooks.append(model.model.norm.register_forward_pre_hook(lambda _, a: out.update(r=a[0][idx])))
        model.model(input_ids=x, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return out["r"], out["m"], out["a"]


@torch.inference_mode()
def scan(model, tokenizer, batch_starts=2):
    if batch_starts < 1:
        raise ValueError("batch_starts must be positive")
    if model.config.model_type not in {"llama", "qwen2", "qwen3"}:
        raise ValueError("Requires a dense Llama, Qwen2 or Qwen3 causal LM")
    if any(p.is_meta or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("All model parameters must be materialized in FP32")
    for p in model.parameters():
        if any(not block.isfinite().all() for block in p.detach().reshape(-1).split(1_000_000)):
            raise ValueError("Non-finite model weights; no decision")
    model.eval()
    mlp, norm, head = model.model.layers[-1].mlp, model.model.norm, model.lm_head
    gain, down, rows = norm.weight, mlp.down_proj.weight, head.weight
    v4, starts = lens(mlp, gain, rows)
    mean_row = rows.mean(0)
    contexts = [tokenizer.encode(s + "\n", add_special_tokens=False) for s in CONTEXTS]
    tokens = {s: [s] for s in starts}
    traces = {s: [] for s in starts}
    active = starts
    for _ in range(5):
        following = []
        for i in range(0, len(active), batch_starts):
            group = active[i:i + batch_starts]
            g, c = len(group), len(contexts)
            r, m, a = states(model, [prefix + tokens[s] for s in group for prefix in contexts])
            full = head(norm(r)).log_softmax(-1)
            delta = (full - head(norm(r - m)).log_softmax(-1)).reshape(g, c, -1).median(1).values
            logp = full.reshape(g, c, -1).median(1).values
            eligible = delta.masked_fill(logp < -3, -math.inf)
            for j, s in enumerate(group):
                eligible[j, tokens[s]] = -math.inf
            values, choices = eligible.topk(8, dim=-1)
            writes = ((rows[choices] - mean_row) * gain) @ down
            contrib = (a.reshape(g, c, 1, -1) * writes[:, None] /
                       r.square().mean(-1).sqrt().reshape(g, c, 1, 1))
            positive = contrib.clamp_min(0).sum(-1).clamp_min(1e-9)
            share = (contrib.topk(4, dim=-1).values.sum(-1) / positive).median(1).values
            picks = torch.where(values.isfinite(), values.clamp_min(0) * share, -1.).argmax(-1)
            for j, s in enumerate(group):
                k = int(picks[j])
                w, token = float(values[j, k]), int(choices[j, k])
                if math.isfinite(w) and w >= 1:
                    traces[s].append(tuple(round(x, 3) for x in (w, float(share[j, k]), float(logp[j, token]))))
                    tokens[s].append(token)
                    following.append(s)
        active = following
        if not active:
            break
    ranking = sorted(((max((d * f for d, f, _ in t), default=0), s)
                      for s, t in traces.items()), reverse=True)[:300]
    return dict(starts=len(starts), **score(v4, (traces[s] for strength, s in ranking if strength > 0)))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-starts", type=int, default=2)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        parser.error("Supply an existing local checkpoint directory")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32, use_safetensors=True,
                                                 local_files_only=True, trust_remote_code=False)
    if any(p.is_meta for p in model.parameters()):
        parser.error("Loader left unmaterialized weights; no result")
    print(json.dumps(scan(model.to(args.device), tok, args.batch_starts), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
