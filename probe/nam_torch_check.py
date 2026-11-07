#!/usr/bin/env python3
"""Check fun/nam_amp.py's float32 reference against the official neural-amp-modeler torch code (not run by Claude:
the sandbox refuses to execute downloaded packages). Needs the wheel unzipped somewhere:

    pip download neural-amp-modeler --no-deps -d /tmp/namdl && unzip -q /tmp/namdl/neural_amp_modeler-*.whl -d /tmp/namwheel
    ./env.sh python probe/nam_torch_check.py /tmp/namwheel ~/models/nam/nam-plugin-reaper/model.nam [more.nam ...]

Prints max and rms difference between the two forward passes on a 2 s sweep + noise signal; expect < 1e-5.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fun"))
from nam_amp import NAM, RefFP32, test_signal
from nam.models._from_nam import init_from_nam


def convert(cfg):
    """0.13.0 wants a 'head' dict per layer array; legacy files have head_size / head_bias"""
    cfg = json.loads(json.dumps(cfg))
    for lc in cfg["config"]["layers"]:
        if "head" not in lc:
            lc["head"] = {"out_channels": lc.pop("head_size"), "kernel_size": 1, "bias": lc.pop("head_bias")}
    return cfg


for path in sys.argv[2:]:
    model = init_from_nam(convert(json.load(open(path)))).eval()
    m = NAM(path); x = test_signal(2.0)
    with torch.no_grad():
        yt = model(torch.tensor(x)[None, None, :]).numpy().reshape(-1)
    y = RefFP32(m).process(x)[m.receptive - 1:]           # torch returns the 'valid' part, aligned to the end
    n = min(len(yt), len(y)); err = np.abs(yt[:n] - y[:n])
    print(f"{os.path.basename(path)}: max |diff| {err.max():.3e}, rms {np.sqrt((err ** 2).mean()):.3e}, peak {np.abs(yt).max():.3e}")
