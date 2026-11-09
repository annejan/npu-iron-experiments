#!/usr/bin/env python3
"""Train a new style for style_cam.py: Johnson, Alahi and Fei-Fei's fast style transfer (the pytorch/examples recipe),
on the CPU with torch, written as an ONNX file with the same graph and weight names as the ONNX model zoo's five, so
style_cam.py and style_tray.py take it as it is (`--model`, or a sixth style in the tray).

  ./env.sh python fun/style_train.py --style ~/models/style/vangogh/starry-night-1280.jpg \
      --out ~/models/style/vangogh/starry-night-9.onnx --steps 8000
  ./env.sh python fun/style_train.py --bench                      # time a few steps (threads, bf16)

Loss: VGG16 (ImageNet weights, ~/models/vgg/vgg16-397923af.pth from download.pytorch.org) content loss at relu2_2 and
Gram-matrix style loss at relu1_2, relu2_2, relu3_3 and relu4_3, content weight 1e5, style weight 1e10, Adam 1e-3,
batches of 4 crops of 256x256 (as pytorch/examples/fast_neural_style). Content images: COCO val2017 (5000 pictures,
~/models/coco/val2017), cached once as 256x256 crops.

The network: 9x9 conv to 32 channels, two stride-2 3x3 convs to 128, five residual blocks, two nearest-upsample 3x3
convs and a 9x9 conv to RGB, reflection padding, instance norm + ReLU after every conv but the last. Input and output
are 0..255. The ONNX template is one of the zoo files; only the weights are replaced. A checkpoint (.pt, weights only)
is kept next to the output so a run can be resumed (`--resume`). `--style-crop x0,y0,x1,y1` (fractions) trains on part
of the painting: only the sky of The Starry Night, at `--style-size 256`, gives much bigger swirls.

Downloads (by hand, once):
  https://download.pytorch.org/models/vgg16-397923af.pth                  -> ~/models/vgg/
  http://images.cocodataset.org/zips/val2017.zip                          -> unzip in ~/models/coco/
  the painting, e.g. a 1280 px thumbnail from Wikimedia Commons (Google Art Project scans of public domain works)
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np

HOME = Path.home()
MODELS = HOME / "models"
NAMES = ("conv1", "in1", "conv2", "in2", "conv3", "in3", *[f"res{r}.{p}" for r in range(1, 6) for p in ("conv1", "in1", "conv2", "in2")],
         "deconv1", "in4", "deconv2", "in5", "deconv3")


def build():
    import torch, torch.nn as nn, torch.nn.functional as F

    class ConvLayer(nn.Module):
        def __init__(s, cin, cout, k, stride=1, up=1):
            super().__init__(); s.k, s.stride, s.up = k, stride, up
            s.conv2d = nn.Conv2d(cin, cout, k, stride)

        def forward(s, x):
            if s.up > 1: x = F.interpolate(x, scale_factor=s.up, mode="nearest")
            return s.conv2d(F.pad(x, (s.k // 2,) * 4, mode="reflect"))

    class Res(nn.Module):
        def __init__(s, c):
            super().__init__()
            s.conv1 = ConvLayer(c, c, 3); s.in1 = nn.InstanceNorm2d(c, affine=True)
            s.conv2 = ConvLayer(c, c, 3); s.in2 = nn.InstanceNorm2d(c, affine=True)

        def forward(s, x):
            return x + s.in2(s.conv2(F.relu(s.in1(s.conv1(x)))))

    class TransformerNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.conv1 = ConvLayer(3, 32, 9); s.in1 = nn.InstanceNorm2d(32, affine=True)
            s.conv2 = ConvLayer(32, 64, 3, 2); s.in2 = nn.InstanceNorm2d(64, affine=True)
            s.conv3 = ConvLayer(64, 128, 3, 2); s.in3 = nn.InstanceNorm2d(128, affine=True)
            for r in range(1, 6): setattr(s, f"res{r}", Res(128))
            s.deconv1 = ConvLayer(128, 64, 3, up=2); s.in4 = nn.InstanceNorm2d(64, affine=True)
            s.deconv2 = ConvLayer(64, 32, 3, up=2); s.in5 = nn.InstanceNorm2d(32, affine=True)
            s.deconv3 = ConvLayer(32, 3, 9)

        def forward(s, x):
            y = F.relu(s.in1(s.conv1(x))); y = F.relu(s.in2(s.conv2(y))); y = F.relu(s.in3(s.conv3(y)))
            for r in range(1, 6): y = getattr(s, f"res{r}")(y)
            y = F.relu(s.in4(s.deconv1(y))); y = F.relu(s.in5(s.deconv2(y)))
            return s.deconv3(y)

    class VGG16(nn.Module):
        """torchvision's vgg16().features[:23] (up to relu4_3), same indices so the state dict loads as it is."""
        def __init__(s):
            super().__init__()
            cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
            layers, c = [], 3
            for v in cfg:
                if v == "M": layers.append(nn.MaxPool2d(2, 2))
                else: layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]; c = v
            s.features = nn.Sequential(*layers)
            s.taps = (3, 8, 15, 22)                          # relu1_2, relu2_2, relu3_3, relu4_3

        def forward(s, x):
            out = []
            for i, l in enumerate(s.features):
                x = l(x)
                if i in s.taps: out.append(x)
            return out

    return TransformerNet, VGG16


def load_vgg(VGG16, path):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    v = VGG16()
    v.load_state_dict({k: t for k, t in sd.items() if k.startswith("features.") and int(k.split(".")[1]) < 23})
    for p in v.parameters(): p.requires_grad_(False)
    return v.eval()


def content_cache(data, size, limit=None):
    """COCO pictures as size x size crops (shorter side scaled to size, centre crop), cached in one .npy."""
    import cv2
    files = sorted(Path(data).glob("*.jpg"))[:limit]
    assert files, f"no .jpg in {data}"
    cache = Path(data).parent / f"{Path(data).name}_{size}_{len(files)}.npy"
    if cache.exists(): return np.load(cache, mmap_mode="r")
    from concurrent.futures import ThreadPoolExecutor

    def one(f):
        im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB); h, w = im.shape[:2]; s = size / min(h, w)
        im = cv2.resize(im, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]; y, x = (h - size) // 2, (w - size) // 2
        return im[y:y + size, x:x + size]
    with ThreadPoolExecutor(8) as ex: arr = np.stack(list(ex.map(one, files)))
    np.save(cache, arr); print(f"cached {len(arr)} content crops in {cache}")
    return arr


def load_rgb(path, long_side=None, crop=None):
    """crop: x0,y0,x1,y1 as fractions of the picture, applied before the scaling."""
    import cv2
    im = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    if crop:
        x0, y0, x1, y1 = crop; h, w = im.shape[:2]
        im = im[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)]
    if long_side:
        s = long_side / max(im.shape[:2]); im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return im


def to_t(rgb):
    import torch
    return torch.from_numpy(np.ascontiguousarray(np.asarray(rgb, np.float32).transpose(2, 0, 1)))[None]


def from_t(y):
    return np.clip(y[0].detach().float().numpy().transpose(1, 2, 0) + 0.5, 0, 255).astype(np.uint8)


def export_onnx(model, template, out):
    """The template zoo graph with this model's weights: same names, same shapes."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(template)
    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    seen = 0
    for i, t in enumerate(m.graph.initializer):
        if t.name in sd:
            a = sd[t.name]; assert tuple(a.shape) == tuple(t.dims), (t.name, a.shape, t.dims)
            m.graph.initializer[i].CopyFrom(numpy_helper.from_array(a.astype(np.float32), t.name)); seen += 1
    assert seen == len(sd), f"{len(sd) - seen} weights not in the template"
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp"); onnx.save(m, tmp); os.replace(tmp, out)
    for q in out.parent.glob(out.stem + ".quant*.npz"): q.unlink()       # stale int8 calibration of an older version


def load_onnx_weights(model, path):
    import onnx, torch
    from onnx import numpy_helper
    w = {t.name: numpy_helper.to_array(t) for t in onnx.load(path).graph.initializer}
    model.load_state_dict({k: torch.from_numpy(w[k].copy()) for k in model.state_dict()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", default=str(MODELS / "style/vangogh/starry-night-1280.jpg"))
    ap.add_argument("--style-size", type=int, default=512, help="long side of the style picture (smaller = bigger strokes)")
    ap.add_argument("--style-crop", help="x0,y0,x1,y1 fractions: train on part of the painting (e.g. only the sky's swirls)")
    ap.add_argument("--data", default=str(MODELS / "coco/val2017"))
    ap.add_argument("--vgg", default=str(MODELS / "vgg/vgg16-397923af.pth"))
    ap.add_argument("--template", default=str(MODELS / "style/onnx-zoo/udnie-9.onnx"))
    ap.add_argument("--out", default=str(MODELS / "style/vangogh/starry-night-9.onnx"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--content-weight", type=float, default=1e5)
    ap.add_argument("--style-weight", type=float, default=1e10)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bf16", action="store_true", help="VGG and the network in bfloat16 autocast")
    ap.add_argument("--init", help="start from the weights of this .onnx (e.g. a zoo style) instead of random")
    ap.add_argument("--resume", action="store_true", help="continue from the checkpoint next to --out")
    ap.add_argument("--every", type=int, default=250, help="checkpoint, export and preview every N steps")
    ap.add_argument("--preview", default=str(MODELS / "yolo26n/coco128/images/train2017/000000000030.jpg"))
    ap.add_argument("--preview-dir", help="save a preview PNG every --every steps here")
    ap.add_argument("--bench", action="store_true", help="time 10 steps and exit")
    a = ap.parse_args()

    import torch, torch.nn.functional as F
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    TransformerNet, VGG16 = build()
    net = TransformerNet(); vgg = load_vgg(VGG16, a.vgg)
    out = Path(a.out); ckpt = out.with_suffix(".pt")
    opt = torch.optim.Adam(net.parameters(), a.lr)
    step = 0
    if a.resume and ckpt.exists():
        c = torch.load(ckpt, map_location="cpu", weights_only=True)
        net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); step = int(c["step"]); print(f"resumed at step {step}")
    elif a.init:
        load_onnx_weights(net, a.init); print(f"init from {a.init}")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    norm = lambda x: (x / 255.0 - mean) / std

    def gram(f):
        b, c, h, w = f.shape; f = f.reshape(b, c, h * w).float()
        return f @ f.transpose(1, 2) / (c * h * w)

    with torch.no_grad():
        style_grams = [gram(f) for f in vgg(norm(to_t(load_rgb(a.style, a.style_size, a.style_crop and [float(v) for v in a.style_crop.split(",")]))))]
    data = content_cache(a.data, a.size) if not a.bench else np.random.randint(0, 255, (64, a.size, a.size, 3), np.uint8)
    rng = np.random.default_rng(step)
    prev = load_rgb(a.preview) if a.preview and Path(a.preview).exists() else None
    ac = (lambda: torch.autocast("cpu", torch.bfloat16)) if a.bf16 else (lambda: torch.autocast("cpu", enabled=False))
    print(f"{len(data)} content crops, style {a.style}{' crop ' + a.style_crop if a.style_crop else ''} at {a.style_size}, {a.steps} steps of {a.batch}x{a.size}x{a.size}, "
          f"{a.threads} threads{', bf16' if a.bf16 else ''}", flush=True)

    net.train(); t0 = time.perf_counter(); lc = ls = 0.0; n = 0
    while step < (10 if a.bench else a.steps):
        x = torch.from_numpy(np.ascontiguousarray(data[rng.integers(0, len(data), a.batch)]).astype(np.float32).transpose(0, 3, 1, 2))
        if rng.random() < 0.5: x = x.flip(3)
        with ac():
            y = net(x)
            fy = vgg(norm(y))
            with torch.no_grad(): fx = vgg(norm(x))
            c_loss = a.content_weight * F.mse_loss(fy[1].float(), fx[1].float())
            s_loss = a.style_weight * sum(F.mse_loss(gram(f), g.expand(f.shape[0], -1, -1)) for f, g in zip(fy, style_grams))
        loss = c_loss + s_loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        step += 1; n += 1; lc += c_loss.item(); ls += s_loss.item()
        if a.bench and step == 2: t0 = time.perf_counter(); n = 0
        if not a.bench and (step % 50 == 0 or step == a.steps):
            dt = (time.perf_counter() - t0) / n
            print(f"step {step}: content {lc / n:.3g}, style {ls / n:.3g}, {dt:.2f} s/step, "
                  f"{(a.steps - step) * dt / 3600:.1f} h left", flush=True); t0 = time.perf_counter(); lc = ls = 0.0; n = 0
        if not a.bench and (step % a.every == 0 or step == a.steps):
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt.with_suffix(".tmp"))
            os.replace(ckpt.with_suffix(".tmp"), ckpt)
            export_onnx(net, a.template, out)
            if prev is not None and a.preview_dir:
                import cv2
                net.eval()
                with torch.no_grad(): p = from_t(net(to_t(prev)))
                net.train()
                Path(a.preview_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(a.preview_dir) / f"step{step:05d}.jpg"), cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
    if a.bench:
        print(f"{(time.perf_counter() - t0) / n:.2f} s/step"); return
    # the exported file against this model, through the same numpy reference style_cam.py checks the NPU with
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import style_cam as SC
    ref = SC.Net(out); img = prev if prev is not None else data[0]
    net.eval()
    with torch.no_grad(): yt = net(to_t(img))[0].numpy()
    print(f"wrote {out}: numpy reference vs torch {SC.psnr(SC.to_rgb(ref.forward(img)), SC.to_rgb(yt)):.1f} dB")


if __name__ == "__main__":
    main()
