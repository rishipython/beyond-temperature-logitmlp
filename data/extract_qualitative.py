"""Extract qualitative LogitMLP examples for the blog's 'What is LogitMLP actually learning?' section.

Outputs blog/data/qualitative_examples.json with a curated list of high-KL positions:
  - prompt context (the few tokens before/after the chosen position)
  - the chosen token, base top-k and warped top-k
  - the (s1, s2, b) values at that position
  - per-position entropy/top1 prob/KL

We pick examples that show the interesting "moderate-confidence structural choice point"
flavor described in the analysis report: not maximum-entropy positions, but ones where the
warp meaningfully changes the second-place token's probability.
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path("/home/timothygao/extra_layer_rlvr")
ANALYSIS = ROOT / "analysis_qwen_pl_warp"
OUT = ROOT / "blog" / "data"


def detok(tok, ids):
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def per_token_strings(tok, ids):
    """Decode each id individually so we can show per-token boundaries."""
    return [tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in ids]


def context_window(tok, all_ids, center, before=20, after=3):
    lo = max(0, center - before)
    hi = min(len(all_ids), center + 1 + after)
    pre = detok(tok, all_ids[lo:center])
    cur = detok(tok, [all_ids[center]])
    post = detok(tok, all_ids[center + 1 : hi])
    return pre, cur, post


def top_k(logp, k=8):
    p = torch.softmax(logp, dim=-1)
    vals, ids = torch.topk(p, k)
    return [(int(i), float(v)) for v, i in zip(vals.tolist(), ids.tolist())]


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

    rollouts = torch.load(ANALYSIS / "rollouts.pt", map_location="cpu", weights_only=False)

    # Concatenate completion_ids in the same order they were processed in
    # 02_collect_distributions; the per_pos arrays index by global_pos into this
    # flat sequence.
    cum = [0]
    for r in rollouts:
        cum.append(cum[-1] + len(r["completion_ids"]))

    def find_rollout(global_pos):
        # Returns (rollout_idx, local_pos in completion_ids)
        for i in range(len(rollouts)):
            if cum[i] <= global_pos < cum[i + 1]:
                return i, global_pos - cum[i]
        raise ValueError("global_pos out of range")

    out = {}

    for variant in ("pl_warp_dapo", "pl_warp_passk"):
        data = torch.load(ANALYSIS / "per_variant" / f"{variant}.pt", map_location="cpu", weights_only=False)
        pp = data["per_pos"]
        kl = pp["kl_method_base"]
        entropy_base = pp["entropy_base"]
        top1_base = pp["top1_prob_base"]
        s1 = pp["s1"]
        s2 = pp["s2"]
        b = pp["b"]

        # Build a per-record map for fast lookup.
        records = {r["global_pos"]: r for r in data["full_dist_records"]}

        # Pick interesting positions: moderate-confidence branching points with
        # meaningful warp activity. We exclude two failure modes:
        #   (a) near-deterministic 2-candidate positions (e.g. choice between
        #       "newline" and "<|im_end|>"), which are dramatic in KL units
        #       but visually uninformative as branching examples.
        #   (b) max-entropy positions where the base distribution is so flat
        #       that everything is roughly equally likely.
        # We use entropy_base \in [0.6, 2.5] nats as a rough proxy for
        # "moderate-confidence structural choice point."
        n = len(kl)
        idxs = torch.arange(n)
        mask_moderate = (
            (top1_base > 0.30) & (top1_base < 0.80)
            & (entropy_base > 0.6) & (entropy_base < 2.5)
            & (kl > 0.05)
        )
        candidates = idxs[mask_moderate]
        # Rank candidates by KL; show the top ~12 distinct rollout-source ones.
        sorted_by_kl = candidates[kl[candidates].argsort(descending=True)]

        examples = []
        seen_prompts = set()
        for gp in sorted_by_kl.tolist():
            try:
                rid, lpos = find_rollout(gp)
            except ValueError:
                continue
            r = rollouts[rid]
            # one example per prompt to maintain diversity
            if r["prompt_text"] in seen_prompts:
                continue

            rec = records.get(gp)
            if rec is None:
                continue

            base_top = top_k(rec["logp_base"], k=6)
            warp_top = top_k(rec["logp_method"], k=6)

            # Skip degenerate "binary choice" positions (top-2 covers most mass).
            top2_mass = base_top[0][1] + base_top[1][1]
            if top2_mass > 0.94:
                continue

            # Filter to examples that match the variant's headline direction so
            # the side-by-side comparison cleanly shows the claimed effect.
            # DAPO is supposed to flatten at structural choice points; pass@k
            # is supposed to sharpen at them.
            base_top1 = base_top[0][1]
            warp_top1 = warp_top[0][1]
            if variant == "pl_warp_dapo":
                if warp_top1 >= base_top1 - 0.05:
                    continue  # not flattening
            else:  # pl_warp_passk
                if warp_top1 <= base_top1 + 0.10:
                    continue  # not meaningfully sharpening

            pre, cur, post = context_window(tok, r["completion_ids"], lpos, before=12, after=2)

            base_top_dec = [(tok.decode([i], skip_special_tokens=False), p) for i, p in base_top]
            warp_top_dec = [(tok.decode([i], skip_special_tokens=False), p) for i, p in warp_top]

            examples.append({
                "global_pos": int(gp),
                "prompt": r["prompt_text"].split("user\n", 1)[-1].split("<|im_end|>", 1)[0].strip()[:500],
                "pre": pre,
                "chosen": cur,
                "post": post,
                "kl": float(kl[gp]),
                "entropy_base": float(entropy_base[gp]),
                "top1_base": float(top1_base[gp]),
                "s1": float(s1[gp]),
                "s2": float(s2[gp]),
                "b": float(b[gp]),
                "base_top": base_top_dec,
                "warp_top": warp_top_dec,
            })
            seen_prompts.add(r["prompt_text"])
            if len(examples) >= 12:
                break

        out[variant] = examples
        print(f"{variant}: {len(examples)} examples")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "qualitative_examples.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT / 'qualitative_examples.json'}")


if __name__ == "__main__":
    main()
