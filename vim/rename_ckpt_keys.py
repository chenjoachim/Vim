"""
rename_ckpt_keys.py
--------------------
Renames all state-dict keys that contain the legacy module name 'bin_mask'
to the new name 'score_predictors' in a PyTorch checkpoint.

Usage:
    python rename_ckpt_keys.py --input <path/to/old.pth> --output <path/to/new.pth>
    python rename_ckpt_keys.py --input <path/to/old.pth>          # overwrites in-place
"""

import argparse
import torch


OLD_NAME = "bin_masks"
NEW_NAME = "score_predictors"


def rename_keys(state_dict: dict) -> tuple[dict, int]:
    """Return a new state_dict with renamed keys and the count of renames."""
    new_sd = {}
    count = 0
    for k, v in state_dict.items():
        new_k = k.replace(OLD_NAME, NEW_NAME)
        if new_k != k:
            print(f"  {k!r:60s}  ->  {new_k!r}")
            count += 1
        new_sd[new_k] = v
    return new_sd, count


def main():
    parser = argparse.ArgumentParser(description="Rename checkpoint module keys.")
    parser.add_argument("--input",  required=True, help="Path to the source checkpoint (.pth).")
    parser.add_argument("--output", default=None,  help="Path for the output checkpoint. Defaults to overwriting --input.")
    args = parser.parse_args()

    out_path = args.output or args.input

    print(f"Loading checkpoint: {args.input}")
    ckpt = torch.load(args.input, map_location="cpu")

    # Support both raw state-dicts and wrapped checkpoints (e.g. {'model': sd, ...})
    if isinstance(ckpt, dict) and "model" in ckpt:
        print("Detected wrapped checkpoint (key='model').")
        new_sd, count = rename_keys(ckpt["model"])
        ckpt["model"] = new_sd
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        print("Detected wrapped checkpoint (key='state_dict').")
        new_sd, count = rename_keys(ckpt["state_dict"])
        ckpt["state_dict"] = new_sd
    else:
        print("Treating checkpoint as a raw state dict.")
        ckpt, count = rename_keys(ckpt)

    if count == 0:
        print(f"\nNo keys containing '{OLD_NAME}' were found — nothing to rename.")
    else:
        print(f"\nRenamed {count} key(s): '{OLD_NAME}' -> '{NEW_NAME}'")
        torch.save(ckpt, out_path)
        print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
