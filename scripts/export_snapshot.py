#!/usr/bin/env python3
"""Export a rolling per-epoch training snapshot to a .cact deployment blob.

Usage:
    python3 scripts/export_snapshot.py /tmp/full_epoch_snapshot.pkl \
        /tmp/robot_epoch.cact [--bits-map "spec"]

Snapshot format matches train_full.py's --snapshot output:
    {"params": ..., "config": {...}, "meta": {...}}
"""
import argparse

import jax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--bits-map", default=None)
    ap.add_argument("--bits", type=int, default=4)
    args = ap.parse_args()

    import pickle

    from needle.model.architecture import TransformerConfig
    from needle.model.export import write_export
    from needle.model.tokenizer import get_tokenizer

    with open(args.snapshot, "rb") as fh:
        ckpt = pickle.load(fh)
    params, cfg = ckpt["params"], TransformerConfig(**ckpt["config"])
    tok = get_tokenizer(cfg.vocab_size)
    info = write_export(jax.tree.map(lambda a: jax.numpy.asarray(a), params),
                        cfg, args.out,
                        bits=args.bits if args.bits_map is None else 4,
                        bits_map=args.bits_map, tokenizer=tok)
    print(f"exported {info['path']}  {info['bytes'] / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
