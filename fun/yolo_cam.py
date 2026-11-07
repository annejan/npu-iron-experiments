#!/usr/bin/env python3
"""YOLO26n object detection on the NPU from the webcam (or an image file), using the stock mlir-aie example.

  ../env.sh python fun/yolo_cam.py --image some.jpg        one image -> yolo_cam.png with boxes
  ../env.sh python fun/yolo_cam.py --still                 one webcam frame -> yolo_cam.png
  ../env.sh python fun/yolo_cam.py --window                live pygame window (ESC)
  ... --synthetic 7                                        seeded random weights (no checkpoint; boxes are noise)

Needs the exported, quantized weights: see mlir-aie/programming_examples/ml/yolo26n/README.md (`make export`,
`make data` with IRON_EXAMPLE_WEIGHTS_DIR set). The whole network (168 int8 operators, 30 workers + 2 value
tiles, every one of the 32 tiles) runs in one dispatch from a letterboxed 640x640 RGB image to (300, 6)
detections (x1, y1, x2, y2, score, class). This script only adds the camera, the letterbox and the drawing.
"""
import argparse, hashlib, os, sys, time
from pathlib import Path
import numpy as np

EX = Path(__file__).resolve().parent.parent / "mlir-aie" / "programming_examples" / "ml" / "yolo26n"
sys.path.insert(0, str(EX))
os.chdir(EX)                          # the example resolves data/ relative to its own directory
import aie.iron as iron
from aie.iron.device import NPU2
import yolo26n as Y
from yolo_params import DATA_DIR, Params, synthetic_params
from yolo_spec import IMAGE_SIZE, MAX_DETECTIONS, Network

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic-light fire-hydrant stop-sign parking-meter bench "
        "bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard "
        "sports-ball kite baseball-bat baseball-glove skateboard surfboard tennis-racket bottle wine-glass cup fork knife spoon "
        "bowl banana apple sandwich orange broccoli carrot hot-dog pizza donut cake chair couch potted-plant bed dining-table "
        "toilet tv laptop mouse remote keyboard cell-phone microwave oven toaster sink refrigerator book clock vase scissors "
        "teddy-bear hair-drier toothbrush").split()


def letterbox_array(bgr, size=IMAGE_SIZE, pad=114):
    """Ultralytics LetterBox on an in-memory BGR frame -> (RGB uint8 square, (ratio, left, top))"""
    import cv2
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), pad, np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    out[top:top + nh, left:left + nw] = resized
    return out[:, :, ::-1].copy(), (r, left, top)


class Detector:
    def __init__(self, synthetic=None, batch=1):
        iron.set_current_device(NPU2())
        self.network = Network.yolo26n()
        if synthetic is not None:
            params_path = synthetic_params(synthetic, DATA_DIR / f"yolo26n_params_synthetic{synthetic}.npz")
        else:
            params_path = DATA_DIR / "yolo26n_params.npz"
            if not params_path.exists():
                sys.exit(f"{params_path} missing: run `make export` and `make data` in {EX} first (see its README)")
        params = Params.load(params_path)
        self.plan = Y.Plan(self.network, params)
        self.batch = batch
        capacity = max(Y.BATCHES)
        self.kwargs = dict(batches=Y.BATCHES, params_path=str(params_path),
                           params_digest=hashlib.sha1(params_path.read_bytes()).hexdigest(), ahead=3)
        self.image_tensor = iron.tensor((capacity * Y.IMAGE_WORDS,), dtype=np.int32)
        self.scratch = iron.tensor((capacity * self.plan.words,), dtype=np.int32)
        self.detections = iron.tensor((capacity * Y.DETECTION_WORDS,), dtype=np.int32)
        self.batch_image = self.image_tensor.subview(0, (batch * Y.IMAGE_WORDS,))
        self.scratch.subview(0, (batch * self.plan.words,)).fill_(0)
        self.batch_detections = self.detections.subview(0, (batch * Y.DETECTION_WORDS,))
        self.batch_detections.fill_(0)
        self.weights = iron.tensor(self.plan.weights.view(np.int32))
        self.ms = 0.0

    def run(self, rgb640):
        with self.batch_image.overwrite() as words:
            words[:] = rgb640.reshape(-1).view(np.int32)
        t0 = time.perf_counter()
        Y.yolo26n(self.image_tensor, self.scratch, self.detections, self.weights, entry=f"b{self.batch}", **self.kwargs)
        self.ms = (time.perf_counter() - t0) * 1e3
        det = self.batch_detections.numpy().reshape(self.batch, MAX_DETECTIONS, 6).view(np.float32)[0]
        return det


def draw(bgr, det, fit, thresh=0.25):
    import cv2
    r, left, top = fit
    n = 0
    for x1, y1, x2, y2, score, cls in det:
        if score < thresh:
            continue
        n += 1
        p1 = (int((x1 - left) / r), int((y1 - top) / r)); p2 = (int((x2 - left) / r), int((y2 - top) / r))
        cv2.rectangle(bgr, p1, p2, (0, 255, 120), 2)
        name = COCO[int(cls)] if 0 <= int(cls) < len(COCO) else str(int(cls))
        cv2.putText(bgr, f"{name} {score:.2f}", (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--synthetic", type=int)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.25)
    a = ap.parse_args()
    import cv2
    det = Detector(a.synthetic)
    out_png = str(Path(__file__).resolve().parent.parent / "yolo_cam.png")
    if a.image:
        bgr = cv2.imread(a.image)
        rgb, fit = letterbox_array(bgr)
        d = det.run(rgb); d = det.run(rgb)
        n = draw(bgr, d, fit, a.thresh)
        cv2.imwrite(out_png, bgr)
        print(f"{a.image}: {n} detections >= {a.thresh} (NPU {det.ms:.1f} ms); wrote yolo_cam.png")
        for x1, y1, x2, y2, s, c in sorted(d, key=lambda t: -t[4])[:8]:
            if s >= a.thresh: print(f"   {COCO[int(c)]:14s} {s:.2f}  [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        return
    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit("no camera")
    if a.still or not a.window:
        for _ in range(5): ok, bgr = cap.read()        # let exposure settle
        rgb, fit = letterbox_array(bgr)
        d = det.run(rgb); d = det.run(rgb)
        n = draw(bgr, d, fit, a.thresh)
        cv2.imwrite(out_png, bgr)
        print(f"webcam frame {bgr.shape[1]}x{bgr.shape[0]}: {n} detections >= {a.thresh}; NPU {det.ms:.1f} ms per image; wrote yolo_cam.png")
        return
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    font = pygame.font.SysFont(None, 26)
    n, t0 = 0, time.perf_counter()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb, fit = letterbox_array(bgr)
        d = det.run(rgb)
        k = draw(bgr, d, fit, a.thresh)
        n += 1
        frame = cv2.resize(bgr, (1280, 720))[:, :, ::-1]
        screen.blit(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (0, 0))
        screen.blit(font.render(f"YOLO26n on the NPU: {det.ms:.1f} ms/image  {n / (time.perf_counter() - t0):.1f} fps  {k} objects", True, (255, 255, 0)), (8, 8))
        pygame.display.flip()


if __name__ == "__main__":
    main()
