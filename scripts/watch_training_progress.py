"""Live progress monitor for SAM2PATH Lightning runs.

Run from a second terminal while training is active:
    python scripts/watch_training_progress.py --run-prefix sam2uni2_s0_distance3ch_short
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from datetime import datetime


def latest_event(prefix: str):
    pattern = os.path.join(
        "outputs", "glomeruli", "run", f"{prefix}*", "Logs", "*",
        "sam2path", "version_0", "events.out.tfevents.*",
    )
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def read_scalars(path: str):
    try:
        from tensorboard.backend.event_processing import event_accumulator as ea

        acc = ea.EventAccumulator(path, size_guidance={ea.SCALARS: 0})
        acc.Reload()
        values = {}
        for tag in acc.Tags().get("scalars", []):
            events = acc.Scalars(tag)
            if events:
                event = events[-1]
                values[tag] = (event.step, event.value)
        return values
    except Exception as exc:  # event file can be mid-write
        return {"_read_error": (0, str(exc))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", default="sam2uni2_s0_distance3ch_short")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()

    print(f"Watching outputs/glomeruli/run/{args.run_prefix}* (Ctrl+C to stop)")
    while True:
        event_file = latest_event(args.run_prefix)
        now = datetime.now().strftime("%H:%M:%S")
        if event_file is None:
            print(f"[{now}] waiting for event file ...", flush=True)
        else:
            values = read_scalars(event_file)
            interesting = {
                key: value for key, value in values.items()
                if any(word in key.lower() for word in
                       ("loss", "dice", "iou", "epoch", "metric", "boundary"))
            }
            stat = os.stat(event_file)
            print(
                f"[{now}] event={os.path.basename(event_file)} "
                f"size={stat.st_size / 1024:.1f}KB "
                f"updated={datetime.fromtimestamp(stat.st_mtime).strftime('%H:%M:%S')} "
                f"scalars={interesting or 'none yet'}",
                flush=True,
            )
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
