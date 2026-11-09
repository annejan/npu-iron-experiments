#!/usr/bin/env python3
"""The NPU style camera as a tray icon: a virtual webcam ("NPU Style", v4l2loopback) whose picture the NPU paints
(fun/style_cam.py), controlled from the system tray.

The webcam and the NPU run only while an app (Jitsi, Meet, Zoom, OBS, a browser tab) streams from the virtual camera,
or while the preview window is open. In between, the camera light is off, the NPU idles, and the loopback shows a still
card so apps keep listing the camera. v4l2loopback tells the writer when a reader starts and stops streaming.

  python3 fun/style_tray.py                 tray icon (a Python with PyQt6, e.g. the system one; the NPU part runs
                                            in ironenv through ./env.sh)
  python3 fun/style_tray.py --vcam /dev/video10 --camera 0

Menu: the five styles (plus any in ~/models/style/*/*-9.onnx, see style_train.py) or the plain camera, cycle the styles, "Only while an app uses the camera" (off = always live),
preview window, stop/start, quit. Left click on the icon toggles the preview, middle click picks the next style. The
icon shows the live picture. The choices are remembered (QSettings).

Needs the v4l2loopback module, for example:
  sudo modprobe v4l2loopback video_nr=10 card_label="NPU Style" exclusive_caps=1
(openSUSE's v4l2loopback-autoload loads it at boot as "OBS Virtual Camera" on the first free /dev/video*; --vcam auto,
the default, finds either.)

Two processes: this one (Qt) and a child in ironenv (`--child`) that owns the webcam, the NPU and the loopback. They talk
in JSON lines over the child's stdin and stdout. Frames for the icon and the preview go through a small shared file in
$XDG_RUNTIME_DIR (a sequence counter around each frame, so a torn read is skipped).
"""
import argparse, json, mmap, os, signal, struct, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAMES = {"mosaic": "Mosaic", "candy": "Candy", "rain-princess": "Rain princess", "udnie": "Udnie", "pointilism": "Pointillism"}
W, H = 320, 240
HDR = 16                                                                       # shared frame: u64 seq, u32 w, u32 h
CYCLE_S = 20




def title(path):
    """Menu name of a style file: the zoo names as they are spelled, others from the file name (starry-night-9.onnx)."""
    n = Path(path).stem.removesuffix("-9")
    return NAMES.get(n, n.replace("-", " ").replace("_", " ").capitalize())


# --------------------------------------------------------------------------------------------- child (ironenv side)
def say(**kw):
    print(json.dumps(kw), flush=True)


class SharedFrame:
    """Latest output frame for the tray (seqlock: odd sequence while writing)."""
    def __init__(self, path, w, h):
        self.path, self.w, self.h, self.size, self.seq = path, w, h, HDR + w * h * 3, 0
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        os.ftruncate(fd, self.size); self.mm = mmap.mmap(fd, self.size); os.close(fd)

    def put(self, rgb):
        self.seq += 1; struct.pack_into("<QII", self.mm, 0, self.seq, self.w, self.h)
        self.mm[HDR:self.size] = rgb.tobytes()
        self.seq += 1; struct.pack_into("<Q", self.mm, 0, self.seq)

    def close(self):
        self.mm.close()
        try: os.unlink(self.path)
        except OSError: pass


class Camera:
    """The webcam on its own thread, which also opens it (that takes a second or two: apps keep getting the card
    meanwhile). index None: the first capture node that is not a loopback (nor `exclude`), looked up on this thread
    too. get() returns the newest frame, centre-cropped and scaled to W x H (RGB)."""
    def __init__(self, index, exclude=None):
        self.index, self.exclude, self.cv, self.frame, self.stop, self.started, self.error = index, exclude, threading.Condition(), None, False, False, None
        self.t_open = self.t_ok = time.monotonic()
        self.thread = threading.Thread(target=self.loop, daemon=True); self.thread.start()

    def loop(self):
        import cv2
        cap = None
        try:
            if self.index is None:
                import style_cam as SC
                self.index = SC.find_camera(exclude=self.exclude)
                if self.index is None: self.error = "no webcam found (plug one in?)"; return
            cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
            if not cap.isOpened():
                self.error = f"cannot open camera {self.index} (is another app using it?)"; return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); self.t_ok = time.monotonic()
            while not self.stop:
                ok, bgr = cap.read()
                if not ok: time.sleep(0.02); continue
                h, w = bgr.shape[:2]; s = min(w / W, h / H); cw, ch = int(W * s), int(H * s)
                crop = bgr[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
                rgb = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
                with self.cv: self.frame, self.t_ok, self.started = rgb, time.monotonic(), True; self.cv.notify()
        except Exception as e:                                    # reaches the tray instead of ending the thread quietly
            self.error = f"camera {self.index}: {e}"
        finally:
            if cap is not None: cap.release()                     # here, on this thread: never left inside cap.read()

    def get(self, timeout):
        with self.cv:
            if self.frame is None: self.cv.wait(timeout)
            f, self.frame = self.frame, None
            return f

    def close(self, wait=3.0):
        """Stop the thread; False if it is still inside OpenCV (it releases the camera itself when it returns)."""
        self.stop = True; self.thread.join(timeout=wait)
        return not self.thread.is_alive()


def idle_card(w, h):
    """What apps see while nobody streams: the NPU and the webcam are off."""
    import cv2, numpy as np
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = ((1 - y) * np.array([22, 16, 40], np.float32) + y * np.array([64, 28, 34], np.float32)).repeat(w, 1).astype(np.uint8)
    font, sc = cv2.FONT_HERSHEY_SIMPLEX, w / 640
    for text, size, yy, col in (("NPU Style", 1.6, 0.47, (255, 170, 60)), ("waking up the NPU and the camera ...", 0.6, 0.6, (200, 200, 220))):
        (tw, _), _ = cv2.getTextSize(text, font, size * sc, max(1, int(2 * sc)))
        cv2.putText(img, text, ((w - tw) // 2, int(h * yy)), font, size * sc, col, max(1, int(2 * sc)), cv2.LINE_AA)
    return img


def child(a):
    os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")   # a stalled cap.read() returns in ~1 s, not 10
    sys.path.insert(0, str(HERE))
    import queue
    import style_cam as SC
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.set())   # never inside npu.run
    cmds = queue.Queue()

    def stdin_reader():                                           # the tray's commands; EOF = the tray is gone
        for line in sys.stdin:
            try: cmds.put(json.loads(line))
            except ValueError: pass
        stop.set()
    threading.Thread(target=stdin_reader, daemon=True).start()

    st = {"style": a.style, "cycle": a.cycle, "auto": a.auto, "preview": False}
    say(t="starting")
    npu = SC.StyleNPU(a.model or SC.style_models(), W, H, verbose=False)
    st["style"] = min(max(st["style"], 0), len(npu.nets))
    if stop.is_set(): say(t="bye"); return                       # quit arrived while the NPU part was built
    vw, vh = map(int, a.vcam_size.lower().split("x"))
    vcam = SC.VirtualCam(a.vcam, vw, vh)
    if a.camera is not None and any(p == f"/dev/video{a.camera}" and d == "v4l2 loopback" for p, d, _, _ in SC.v4l2_devices()):
        vcam.close(); sys.exit(f"--camera {a.camera} is a v4l2loopback node, not a webcam")
    shm = SharedFrame(a.shm, W, H) if a.shm else None
    card = vcam.encode(idle_card(vw, vh))                         # once: idle writes then cost no OpenCV work
    say(t="ready", vcam=vcam.dev, camera=a.camera if a.camera is not None else SC.find_camera(exclude=vcam.dev), events=vcam.usage is not None)
    cam, cam_retry, cam_err, old = None, 0.0, None, []           # old: cameras whose thread outlived close()
    frames, t_rate, fps, t_style, t_status = 0, time.monotonic(), 0.0, time.monotonic(), 0.0

    def report(msg):                                              # each distinct error once, until frames flow again
        nonlocal cam_err
        if msg != cam_err: cam_err = msg; say(t="error", msg=msg)

    def drop(c):
        if not c.close(): old.append(c)

    try:
        while not stop.is_set():
            while not cmds.empty():
                c = cmds.get()
                if c.get("quit"): stop.set()
                for k in ("style", "cycle", "auto", "preview"):
                    if k in c: st[k] = c[k]
                if "style" in c or "cycle" in c: t_style = time.monotonic()
            if stop.is_set(): break
            watched = vcam.watched()
            want = (not st["auto"]) or st["preview"] or watched
            now = time.monotonic()
            old[:] = [c for c in old if c.thread.is_alive()]
            if cam is not None and not cam.started and not cam.error and now - cam.t_open > 8:
                cam.error = f"camera {cam.index} opened but delivers no frames"
            if cam is not None and cam.error:
                report(cam.error); drop(cam); cam = None; cam_retry = now + 3
            if want and cam is None and now >= cam_retry and not old:
                cam = Camera(a.camera, exclude=vcam.dev)          # a.camera None: looked up again on every try
            elif not want and cam is not None:
                drop(cam); cam = None
            if cam is None or not cam.started:                    # idle or opening: keep the device listed with the card
                vcam.write(card); frames, t_rate, t_style = 0, now, now   # the rate and the cycle count live time only
                if now - t_status > 1:
                    t_status = now; say(t="status", live=False, watched=watched, style=st["style"], waiting=bool(want))
                vcam.watched(timeout=0.1 if cam else 0.2)
                continue
            rgb = cam.get(0.5)
            if rgb is None:
                if time.monotonic() - cam.t_ok > 3:               # the webcam stopped delivering: reopen it
                    report(f"camera {cam.index} stopped delivering frames"); drop(cam); cam = None; cam_retry = time.monotonic() + 1
                continue
            cam_err = None
            if st["cycle"] and st["style"] > 0 and now - t_style > st["cycle"]:
                st["style"] = st["style"] % len(npu.nets) + 1; t_style = now; say(t="style", style=st["style"])
            out = rgb if st["style"] == 0 else npu.run(rgb, style=st["style"] - 1)
            vcam.write(out)
            if shm: shm.put(out)
            frames += 1
            if now - t_rate >= 1:
                fps, frames, t_rate = frames / (now - t_rate), 0, now
                say(t="status", live=True, watched=watched, style=st["style"], fps=round(fps, 1), ms=round(npu.ms, 2) if st["style"] else 0)
    finally:
        if cam is not None: drop(cam)
        for c in old: c.close(wait=3)
        vcam.close()
        if shm: shm.close()
        say(t="bye")


# ------------------------------------------------------------------------------------------------ tray (Qt side)
def tray(a):
    from PyQt6.QtCore import QProcess, QSettings, Qt, QTimer, QRectF
    from PyQt6.QtGui import QAction, QActionGroup, QColor, QIcon, QImage, QLinearGradient, QPainter, QPainterPath, QPixmap
    from PyQt6.QtWidgets import QApplication, QLabel, QMenu, QSystemTrayIcon

    app = QApplication(sys.argv[:1]); app.setApplicationName("NPU Style"); app.setQuitOnLastWindowClosed(False)
    if not QSystemTrayIcon.isSystemTrayAvailable(): sys.exit("no system tray (StatusNotifier) on this desktop")

    def badge(colors, lens):
        """The idle/stopped icon: a rounded square with a camera lens."""
        pm = QPixmap(64, 64); pm.fill(Qt.GlobalColor.transparent); p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g = QLinearGradient(0, 0, 64, 64); g.setColorAt(0, QColor(colors[0])); g.setColorAt(1, QColor(colors[1]))
        p.setBrush(g); p.setPen(Qt.PenStyle.NoPen); p.drawRoundedRect(QRectF(2, 2, 60, 60), 14, 14)
        p.setBrush(QColor(lens)); p.drawEllipse(QRectF(17, 17, 30, 30))
        p.setBrush(QColor(colors[0])); p.drawEllipse(QRectF(25, 25, 14, 14)); p.end()
        return QIcon(pm)

    def thumb(img):
        """The live icon: the centre of the styled picture with rounded corners."""
        s = min(img.width(), img.height())
        sq = img.copy((img.width() - s) // 2, (img.height() - s) // 2, s, s).scaled(
            64, 64, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        pm = QPixmap(64, 64); pm.fill(Qt.GlobalColor.transparent); p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath(); path.addRoundedRect(QRectF(0, 0, 64, 64), 12, 12); p.setClipPath(path)
        p.drawImage(0, 0, sq); p.end()
        return QIcon(pm)

    class Preview(QLabel):
        def __init__(self, on_close):
            super().__init__(); self.on_close = on_close
            self.setWindowTitle("NPU Style preview"); self.setFixedSize(2 * W, 2 * H)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter); self.setStyleSheet("background: #101018; color: #aaa")
            self.setText("starting the camera ...")

        def closeEvent(self, e):
            self.on_close(); e.accept()

    class Tray:
        def __init__(self):
            self.cfg = QSettings("npu-iron-experiments", "style-tray")
            sys.path.insert(0, str(HERE)); from style_cam import style_models
            self.models = style_models(); self.titles = ["Plain camera"] + [title(p) for p in self.models]   # 0 = no filter
            self.style = min(max(int(self.cfg.value("style", 1)), 0), len(self.models))
            self.auto = self.cfg.value("auto", True, type=bool)
            self.cycle = self.cfg.value("cycle", False, type=bool)
            self.shm = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / f"npu-style-{os.getpid()}.rgb"
            self.mm, self.seen, self.live, self.quitting, self.proc, self.log, self.ready = None, -1, False, False, None, [], False
            self.stopping = False
            self.icons = {"idle": badge(("#ff9a3c", "#7a2bd6"), "#1b1530"), "stopped": badge(("#777", "#444"), "#222")}
            self.icon = QSystemTrayIcon(self.icons["stopped"]); self.icon.activated.connect(self.on_click)
            self.preview = Preview(lambda: self.set_preview(False))
            self.build_menu(); self.icon.show()
            self.t_icon = QTimer(); self.t_icon.timeout.connect(self.update_icon); self.t_icon.start(500)
            self.t_prev = QTimer(); self.t_prev.timeout.connect(self.update_preview)
            self.start()

        def build_menu(self):
            m = self.menu = QMenu()
            self.a_status = m.addAction("starting ..."); self.a_status.setEnabled(False); m.addSeparator()
            grp = QActionGroup(m); grp.setExclusive(True); self.a_style = []
            for i, t in enumerate(self.titles):
                act = QAction(t if i == 0 else f"{i}  {t}", m, checkable=True); act.setChecked(i == self.style)
                act.triggered.connect(lambda _=False, i=i: self.set_style(i)); grp.addAction(act); m.addAction(act); self.a_style.append(act)
                if i == 0: m.addSeparator()
            self.a_cycle = m.addAction(f"Next style every {CYCLE_S} s"); self.a_cycle.setCheckable(True); self.a_cycle.setChecked(self.cycle)
            self.a_cycle.toggled.connect(self.set_cycle); m.addSeparator()
            self.a_auto = m.addAction("Only while an app uses the camera"); self.a_auto.setCheckable(True); self.a_auto.setChecked(self.auto)
            self.a_auto.setToolTip("off: the webcam and the NPU run all the time"); self.a_auto.toggled.connect(self.set_auto)
            self.a_prev = m.addAction("Preview window"); self.a_prev.setCheckable(True); self.a_prev.toggled.connect(self.set_preview)
            m.addSeparator()
            self.a_run = m.addAction("Stop virtual camera"); self.a_run.triggered.connect(self.toggle_run)
            m.addAction("Quit").triggered.connect(self.quit)
            self.icon.setContextMenu(m)

        # -- child process
        def start(self):
            self.ready, self.log, self.stopping = False, [], False
            self.preview.setText("starting the NPU ...")
            p = self.proc = QProcess()
            p.setWorkingDirectory(str(ROOT)); p.setProgram(str(ROOT / "env.sh"))
            args = ["python", str(HERE / "style_tray.py"), "--child", "--shm", str(self.shm), "--style", str(self.style),
                    "--cycle", str(CYCLE_S if self.cycle else 0), "--auto", "1" if self.auto else "0", "--vcam", a.vcam, "--vcam-size", a.vcam_size]
            if a.camera is not None: args += ["--camera", str(a.camera)]
            for m in self.models: args += ["--model", m]
            p.setArguments(args)
            p.readyReadStandardOutput.connect(self.on_stdout); p.readyReadStandardError.connect(self.on_stderr)
            p.finished.connect(self.on_finished); p.errorOccurred.connect(self.on_error); p.start()
            self.a_run.setText("Stop virtual camera"); self.set_status("starting the NPU ...", "stopped")

        def send(self, **kw):
            if self.proc is not None and self.proc.state() == QProcess.ProcessState.Running:
                self.proc.write((json.dumps(kw) + "\n").encode())

        def on_stdout(self):
            while self.proc is not None and self.proc.canReadLine():
                line = bytes(self.proc.readLine()).decode(errors="replace").strip()
                try: msg = json.loads(line)
                except ValueError: self.keep(line); continue
                if isinstance(msg, dict): self.on_msg(msg)

        def on_stderr(self):
            for line in bytes(self.proc.readAllStandardError()).decode(errors="replace").splitlines(): self.keep(line)

        def keep(self, line):
            if line.strip(): self.log = (self.log + [line.strip()])[-30:]

        def on_msg(self, m):
            t = m.get("t")
            if t == "ready":
                self.ready, self.mm, self.seen = True, None, -1
                self.send(style=self.style, cycle=CYCLE_S if self.cycle else 0, auto=self.auto, preview=self.a_prev.isChecked())
                if self.stopping or self.quitting: return
                note = "" if m.get("events") else " (this v4l2loopback cannot report readers: always live)"
                self.icon.showMessage("NPU Style", f"Virtual camera ready on {m['vcam']}. Pick \"NPU Style\" in your video app{note}.",
                                      QSystemTrayIcon.MessageIcon.Information, 4000)
            elif t == "style":
                self.style = m["style"]; self.a_style[self.style].setChecked(True)
            elif t == "status":
                self.live = m.get("live", False)
                name = self.titles[m.get("style", self.style)]
                if self.live:
                    who = "an app is watching" if m.get("watched") else "preview" if self.a_prev.isChecked() else "always on"
                    npu = f", NPU {m['ms']:.1f} ms" if m.get("ms") else ""
                    self.set_status(f"Live: {name}, {m.get('fps', 0):.0f} fps{npu} ({who})", None)
                else:
                    self.set_status("Starting the camera ..." if m.get("waiting") else f"Idle: {name}, waiting for an app", "idle")
            elif t == "error":
                self.icon.showMessage("NPU Style", m.get("msg", "error"), QSystemTrayIcon.MessageIcon.Warning, 6000)
                self.set_status(m.get("msg", "error"), None)

        def on_error(self, err):
            if err == QProcess.ProcessError.FailedToStart:           # no finished signal follows this one
                self.keep(f"cannot start {ROOT / 'env.sh'}: {self.proc.errorString()}"); self.on_finished(1, None)

        def on_finished(self, code, _status):
            self.proc, self.live, self.ready = None, False, False
            self.preview.setText("stopped")
            if self.quitting: QApplication.quit(); return
            why = next((l for l in reversed(self.log) if not l.startswith("{")), "") if code else ""
            self.set_status("Stopped" + (f": {why}" if why else ""), "stopped"); self.a_run.setText("Start virtual camera")
            if code: self.icon.showMessage("NPU Style", f"The NPU part stopped (exit {code}). {why}", QSystemTrayIcon.MessageIcon.Warning, 8000)

        def stop_child(self):
            if self.proc is None: return
            self.stopping = True; self.send(quit=True); proc = self.proc
            QTimer.singleShot(8000, lambda: proc.state() != QProcess.ProcessState.NotRunning and proc.terminate())
            QTimer.singleShot(12000, lambda: proc.state() != QProcess.ProcessState.NotRunning and proc.kill())

        # -- menu actions
        def set_style(self, i):
            self.style = i; self.cfg.setValue("style", i); self.send(style=i)

        def set_cycle(self, on):
            self.cycle = on; self.cfg.setValue("cycle", on); self.send(cycle=CYCLE_S if on else 0)

        def set_auto(self, on):
            self.auto = on; self.cfg.setValue("auto", on); self.send(auto=on)

        def set_preview(self, on):
            if self.a_prev.isChecked() != on: self.a_prev.setChecked(on); return   # re-enters through toggled
            self.send(preview=on)
            if on: self.preview.show(); self.preview.raise_(); self.t_prev.start(33)
            else: self.t_prev.stop(); self.preview.hide()

        def toggle_run(self):
            if self.proc is None: self.start()
            else: self.stop_child(); self.a_run.setText("Stopping ...")

        def on_click(self, reason):
            R = QSystemTrayIcon.ActivationReason
            if reason == R.Trigger: self.set_preview(not self.a_prev.isChecked())
            elif reason == R.MiddleClick:
                i = self.style % len(self.models) + 1; self.a_style[i].setChecked(True); self.set_style(i)

        def quit(self):
            self.quitting = True; self.preview.hide()
            if self.proc is None: QApplication.quit()
            else: self.stop_child()

        # -- pictures
        def set_status(self, text, icon):
            self.a_status.setText(text); self.icon.setToolTip(f"NPU Style\n{text}")
            if icon: self.icon.setIcon(self.icons[icon])

        def frame(self):
            if not self.ready: return None
            if self.mm is None:
                try:
                    with open(self.shm, "rb") as f: self.mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
                except (OSError, ValueError): return None
            s1, w, h = struct.unpack_from("<QII", self.mm, 0)
            if s1 == 0 or s1 == self.seen or s1 % 2 or (w, h) != (W, H) or len(self.mm) < HDR + w * h * 3: return None
            data = self.mm[HDR:HDR + w * h * 3]
            if struct.unpack_from("<Q", self.mm, 0)[0] != s1: return None     # torn: the child wrote meanwhile
            self.seen = s1
            return QImage(data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

        def update_icon(self):
            if not self.live or self.t_prev.isActive(): return    # the preview timer feeds the icon while it runs
            img = self.frame()
            if img is not None: self.icon.setIcon(thumb(img))

        def update_preview(self):
            img = self.frame()
            if img is None: return
            self.preview.setPixmap(QPixmap.fromImage(img).scaled(2 * W, 2 * H, Qt.AspectRatioMode.KeepAspectRatio,
                                                                 Qt.TransformationMode.SmoothTransformation))
            self.n_prev = getattr(self, "n_prev", 0) + 1
            if self.n_prev % 15 == 0 and self.live: self.icon.setIcon(thumb(img))

    t = Tray()
    signal.signal(signal.SIGINT, lambda *_: t.quit())
    keepalive = QTimer(); keepalive.timeout.connect(lambda: None); keepalive.start(200)   # lets Python see Ctrl-C
    if a.seconds:
        QTimer.singleShot(int(a.seconds * 1000), t.quit)
    if a.shot:                                                     # for the README: preview, menu and icon as PNGs
        out = Path(a.shot); out.mkdir(parents=True, exist_ok=True)
        if not a.seconds: QTimer.singleShot(int(a.shot_after * 1000) + 3000, t.quit)
        QTimer.singleShot(1000, lambda: t.set_preview(True))

        def shoot():
            t.preview.grab().save(str(out / "style_tray_preview.png"))
            t.menu.adjustSize(); t.menu.grab().save(str(out / "style_tray_menu.png"))
            t.icon.icon().pixmap(64, 64).save(str(out / "style_tray_icon.png")); print(f"wrote {out}/style_tray_*.png", flush=True)
        QTimer.singleShot(int(a.shot_after * 1000), shoot)
    code = app.exec()
    try: os.unlink(t.shm)
    except OSError: pass
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcam", default="auto", help="v4l2loopback device (default: find it)")
    ap.add_argument("--vcam-size", default="640x480")
    ap.add_argument("--camera", type=int, help="webcam index (default: the first capture device that is not a loopback)")
    ap.add_argument("--seconds", type=float, help="quit after this many seconds (testing)")
    ap.add_argument("--shot", help="directory: open the preview and save preview, menu and icon PNGs, then quit")
    ap.add_argument("--shot-after", type=float, default=12.0)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shm", help=argparse.SUPPRESS)
    ap.add_argument("--model", action="append", help=argparse.SUPPRESS)
    ap.add_argument("--style", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--cycle", type=float, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--auto", type=int, default=1, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child: child(a)
    else: tray(a)


if __name__ == "__main__":
    main()
