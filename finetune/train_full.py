#!/usr/bin/env python3
"""Full-parameter fine-tune of Needle 2 (all weights, pretrained init).

Why: LoRA rank16 margins sit too close to the engine's A8 activation-noise
floor (PLAY-file / UNAVAILABLE / SHOW flip at export). Full FT widens margins;
optional --qat makes weights robust to export quantization so robot.cact can
go back to 2-3 bits (fits 16MB ESP32 flash).

Usage:
  JAX_PLATFORMS=METAL python3 finetune/train_full.py \
      --data data/finetune/train_v2.jsonl \
      --epochs 10 --batch-size 8 --max-len 256 \
      --out checkpoints/full_v1.pkl --cact robot.cact \
      [--qat 4] [--lr 2e-4] [--max-steps N]

Targets use <think></think> (empty reasoning): data records should have
"reasoning": "" — pass --strip-reasoning to blank whatever is in the file.
"""
import argparse, os, pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# jax-metal plugin compat (same trick needle/model/finetune.py uses)
if sys.platform == "darwin":
    os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/finetune/train_v2.jsonl")
    ap.add_argument("--base", default="checkpoints/needle2.pkl")
    ap.add_argument("--init", default="", help="start from a full-FT checkpoint instead of --base")
    ap.add_argument("--out", default="checkpoints/full_v1.pkl")
    ap.add_argument("--cact", default="")
    ap.add_argument("--bits", type=int, default=4, help="export bits for --cact")
    ap.add_argument("--bits-map", default=None,
                    help='per-tensor widths, e.g. "embedding=4,stack/mhc_phi_pre=4,default=2"')
    ap.add_argument("--qat", type=int, default=0, help="train-time weight-quant sim width (0=off)")
    ap.add_argument("--aqat", action="store_true",
                    help="activation+KV+weight QAT via the package's built-in quant path "
                         "(mirrors engine W%dA8/KV8 exactly)" % 4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--snapshot", default="/tmp/full_epoch_snapshot.pkl",
                    help="rolling per-epoch checkpoint (overwritten each epoch)")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--strip-reasoning", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke-test cap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax, jax.numpy as jnp, optax
    from needle.model.run import load_checkpoint
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.tokenizer import get_tokenizer
    from needle.model.finetune import fit_max_len, load_jsonl
    from needle.model.quantize import cq_quantize_params

    t0 = time.time()
    emit = lambda m: print(m, flush=True)

    params, cfg = load_checkpoint(args.base)
    if args.init:
        init = pickle.load(open(args.init, "rb"))["params"]
        params = jax.tree.map(lambda a, b: jnp.asarray(b), params, init, is_leaf=lambda x: x is None or not isinstance(x, dict))
        emit(f"  {'init':<9} {args.init}")
    cfg.dtype = "float32"
    params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
    backend = jax.default_backend().lower()
    if backend == "metal":
        cfg.flash = False
        cfg.remat = False
        cfg.scan_unroll = cfg.num_layers
    emit(f"  {'backend':<9} {backend}  float32  full-param")

    # ---- data
    import json, random
    records = []
    with open(args.data, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.strip_reasoning:
                r["reasoning"] = ""
            elif not r.get("reasoning"):
                r["reasoning"] = ""
            records.append(r)
    tmp = args.data + ".fullft.tmp.jsonl"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    tok = get_tokenizer(cfg.vocab_size)
    max_len = fit_max_len(tmp, tok, args.max_len)
    seqs, masks = load_jsonl(tmp, tok, max_len)
    os.remove(tmp)
    emit(f"  {'data':<9} {len(seqs)} examples  seq_len {max_len}")

    model = SimpleAttentionNetwork(cfg)

    n_val = min(int(len(seqs) * args.val_split), len(seqs) - 1)
    order = np.random.default_rng(args.seed).permutation(len(seqs))
    seqs, masks = seqs[order], masks[order]
    val_seqs, val_masks = seqs[:n_val], masks[:n_val]
    seqs, masks = seqs[n_val:], masks[n_val:]
    emit(f"  {'holdout':<9} {n_val} examples")

    # ---- QAT (STE): forward uses cq-quantized weights, gradients flow straight
    qat_bits = args.qat or 0
    bits_map, map_default = (None, None)
    if args.bits_map:
        from needle.model.quantize import parse_bits_map
        bits_map, map_default = parse_bits_map(args.bits_map)
        if not qat_bits:
            qat_bits = -1          # mixed-width QAT keyed off the map
        emit(f"  {'bits-map':<9} {args.bits_map}")
    if args.aqat:
        from needle.model import quantize as _q
        _q.KV_BITS = 8          # match engine KV-cache width
        _q.ACT_BITS = 8         # engine activation width (default already 8)
        emit(f"  {'aqat':<9} enabled: act A8 + kv8 fake-quant in forward")

    def ste(params_tree):
        if not qat_bits:
            return params_tree
        from needle.model.quantize import _is_quant_leaf, _bits_for, leaf_name
        def fn(path, leaf):
            if not _is_quant_leaf(path, leaf):
                return leaf
            b = (_bits_for(leaf_name(path), bits_map, map_default)
                 if bits_map is not None else qat_bits)
            q = jax.lax.stop_gradient(cq_quantize_params(
                {"w": leaf}, b)["w"])
            return leaf + jax.lax.stop_gradient(q - leaf)
        return jax.tree_util.tree_map_with_path(fn, params_tree)

    def loss_fn(p, ids, mask):
        logits = model.apply({"params": ste(p)}, ids, quant=bool(args.aqat))
        logits, targets, mask = logits[:, :-1], ids[:, 1:], mask[:, 1:]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=max(1, (args.epochs * (-(-len(seqs) // args.batch_size))) // 20),
        decay_steps=args.epochs * (-(-len(seqs) // args.batch_size)))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched))
    opt_state = optimizer.init(params)
    total = args.epochs * (-(-len(seqs) // args.batch_size))
    emit(f"  {'schedule':<9} {total} steps  lr {args.lr:g}  qat {'W%d' % qat_bits if qat_bits else 'off'}  (compiling...)")

    @jax.jit
    def train_step(p, st, ids, mask):
        loss, grads = jax.value_and_grad(loss_fn)(p, ids, mask)
        upd, st = optimizer.update(grads, st, p)
        return optax.apply_updates(p, upd), st, loss

    eval_step = jax.jit(loss_fn)

    every = max(1, total // 50)
    step_i = 0
    for epoch in range(args.epochs):
        perm = np.random.permutation(len(seqs))
        last = 0.0
        for s in range(0, len(seqs), args.batch_size):
            idx = perm[s:s + args.batch_size]
            params, opt_state, loss = train_step(
                params, opt_state, jnp.asarray(seqs[idx]), jnp.asarray(masks[idx]))
            last = float(loss)
            step_i += 1
            if step_i % every == 0:
                emit(f"  {'step':<9} {step_i}/{total}  loss {last:.4f}")
            if args.max_steps and step_i >= args.max_steps:
                break
        val = np.mean([float(eval_step(params, jnp.asarray(val_seqs[i:i + args.batch_size]),
                                       jnp.asarray(val_masks[i:i + args.batch_size])))
                       for i in range(0, n_val, args.batch_size)]) if n_val else float("nan")
        emit(f"  {'epoch':<9} {epoch + 1}/{args.epochs}  loss {last:.4f}  val {val:.4f}  [{time.time()-t0:.0f}s]")
        with open(args.snapshot, "wb") as fh:
            pickle.dump({"params": jax.device_get(params), "config": vars(cfg),
                         "meta": {"base": args.base, "full": True,
                                  "epoch": epoch + 1}}, fh)
        if args.max_steps and step_i >= args.max_steps:
            break

    # ---- save full checkpoint (numpy tree) + export cact
    np_params = jax.tree.map(lambda a: np.asarray(a), params)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"params": np_params, "config": vars(cfg) if hasattr(cfg, "__dict__") else cfg,
                     "base": args.base, "full": True}, fh)
    emit(f"  {'saved':<9} {args.out}")

    if args.cact:
        from needle.model.export import write_export
        from needle.model.architecture import effective_kv_window
        info = write_export(jax.tree.map(lambda a: jnp.asarray(a), np_params), cfg, args.cact,
                            bits=args.bits, bits_map=args.bits_map,
                            tokenizer=tok, kv_window=effective_kv_window(cfg))
        emit(f"  {'wrote':<9} {info['path']}  {info['bytes'] / 1e6:.2f} MB  "
             f"{'map' if args.bits_map else f'W{args.bits}'}A8")
    emit(f"done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
