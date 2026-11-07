#!/usr/bin/env python3
"""npu_top.py - live AMD XDNA NPU telemetry straight from the amdxdna driver.

No root, no XRT: ctypes + fcntl on /dev/accel/accel0 using
DRM_IOCTL_AMDXDNA_GET_INFO (see /usr/include/drm/amdxdna_accel.h).

  python npu_top.py                # dump everything once
  python npu_top.py --watch 0.5    # one line per sample: clocks, power mode, load
"""
import argparse, ctypes as C, errno, fcntl, os, sys, time

DEV = "/dev/accel/accel0"

# --- ioctl number ---------------------------------------------------------
# asm-generic/ioctl.h: _IOC(dir,type,nr,size) = dir<<30 | size<<16 | type<<8 | nr
# DRM_IOWR(nr, T) = _IOWR('d', nr, sizeof T); nr = DRM_COMMAND_BASE(0x40) + idx.
DRM_COMMAND_BASE = 0x40
DRM_AMDXDNA_GET_INFO = 7            # enum amdxdna_drm_ioctl_id

class GetInfo(C.Structure):          # struct amdxdna_drm_get_info, 16 bytes
    _fields_ = [("param", C.c_uint32), ("buffer_size", C.c_uint32), ("buffer", C.c_uint64)]

def _ioc(d, t, nr, size): return (d << 30) | (size << 16) | (ord(t) << 8) | nr
IOCTL_GET_INFO = _ioc(3, "d", DRM_COMMAND_BASE + DRM_AMDXDNA_GET_INFO, C.sizeof(GetInfo))
assert IOCTL_GET_INFO == 0xC0106447

# --- enum amdxdna_drm_get_param -------------------------------------------
Q = {"AIE_STATUS": 0, "AIE_METADATA": 1, "AIE_VERSION": 2, "CLOCK_METADATA": 3,
     "SENSORS": 4, "HW_CONTEXTS": 5, "FIRMWARE_VERSION": 8, "POWER_MODE": 9,
     "TELEMETRY": 10, "FORCE_PREEMPT_STATE": 11, "RESOURCE_INFO": 12,
     "FRAME_BOUNDARY_PREEMPT_STATE": 13}
POWER_MODES = ["DEFAULT", "LOW", "MEDIUM", "HIGH", "TURBO"]
SENSOR_TYPES = {0: "power", 1: "col-util", 2: "temp"}

# --- payload structs (natural alignment matches the uapi layout) ----------
class Clock(C.Structure):
    _fields_ = [("name", C.c_char * 16), ("freq_mhz", C.c_uint32), ("pad", C.c_uint32)]
class ClockMetadata(C.Structure):
    _fields_ = [("mp_npu_clock", Clock), ("h_clock", Clock)]
class AieVersion(C.Structure):
    _fields_ = [("major", C.c_uint32), ("minor", C.c_uint32)]
class TileMeta(C.Structure):
    _fields_ = [(n, C.c_uint16) for n in ("row_count", "row_start", "dma_channel_count",
                                           "lock_count", "event_reg_count")] + [("pad", C.c_uint16 * 3)]
class AieMetadata(C.Structure):
    _fields_ = [("col_size", C.c_uint32), ("cols", C.c_uint16), ("rows", C.c_uint16),
                ("version", AieVersion), ("core", TileMeta), ("mem", TileMeta), ("shim", TileMeta)]
class Sensor(C.Structure):
    _fields_ = [("label", C.c_char * 64), ("input", C.c_uint32), ("max", C.c_uint32),
                ("average", C.c_uint32), ("highest", C.c_uint32), ("status", C.c_char * 64),
                ("units", C.c_char * 16), ("unitm", C.c_int8), ("type", C.c_uint8), ("pad", C.c_uint8 * 6)]
class Hwctx(C.Structure):
    _fields_ = [("context_id", C.c_uint32), ("start_col", C.c_uint32), ("num_col", C.c_uint32),
                ("pad", C.c_uint32), ("pid", C.c_int64)] + \
               [(n, C.c_uint64) for n in ("command_submissions", "command_completions",
                                          "migrations", "preemptions", "errors")]
class FwVersion(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("major", "minor", "patch", "build")]
class PowerMode(C.Structure):
    _fields_ = [("power_mode", C.c_uint8), ("pad", C.c_uint8 * 7)]
class ResourceInfo(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ("npu_clk_max", "npu_tops_max", "npu_task_max",
                                          "npu_tops_curr", "npu_task_curr")]
class AttrState(C.Structure):
    _fields_ = [("state", C.c_uint8), ("pad", C.c_uint8 * 7)]

for cls, sz in ((ClockMetadata, 48), (AieMetadata, 64), (Sensor, 168), (Hwctx, 64),
                (FwVersion, 16), (PowerMode, 8), (ResourceInfo, 40), (AttrState, 8)):
    assert C.sizeof(cls) == sz, (cls.__name__, C.sizeof(cls))

# --- plumbing -------------------------------------------------------------
class QueryError(Exception):
    def __init__(self, name, err):
        self.name, self.err = name, err
        why = {errno.EINVAL: "EINVAL (driver rejected the request/buffer)",
               errno.EOPNOTSUPP: "EOPNOTSUPP (query not supported by this driver/device)",
               errno.EPERM: "EPERM (needs root / CAP_SYS_ADMIN)",
               errno.EACCES: "EACCES (permission denied)",
               errno.ENODEV: "ENODEV (device gone)"}.get(err, "%s" % errno.errorcode.get(err, err))
        super().__init__("%s: %s - %s" % (name, why, os.strerror(err)))

def get_info(fd, name, buf):
    """Run GET_INFO for param `name` into ctypes buffer `buf`; return bytes written back."""
    args = GetInfo(Q[name], C.sizeof(buf), C.addressof(buf))
    try:
        fcntl.ioctl(fd, IOCTL_GET_INFO, args)
    except OSError as e:
        raise QueryError(name, e.errno) from None
    return args.buffer_size

def query_struct(fd, name, cls):
    buf = cls(); get_info(fd, name, buf); return buf

def query_array(fd, name, cls, valid, n=256):
    """Array query. Do not trust the returned buffer_size: the in-tree 7.3 driver
    reports the *leftover* space for HW_CONTEXTS (num_element-- per entry), while
    SENSORS reports bytes used. Count filled entries in the zeroed buffer instead."""
    buf = (cls * n)(); get_info(fd, name, buf)
    return [e for e in buf if valid(e)]

def sensors(fd): return query_array(fd, "SENSORS", Sensor, lambda s: s.label != b"")
def hwctxs(fd): return query_array(fd, "HW_CONTEXTS", Hwctx, lambda h: h.context_id != 0)

def sensor_str(s, raw=False):
    # Driver already labels units (e.g. 'mW') and also sets unitm=-3; show the raw
    # value with its unit string and only mention unitm when it is nonzero.
    t = "%s=%d %s" % (s.label.decode(errors="replace"), s.input, s.units.decode(errors="replace"))
    return t + (" (unitm %d)" % s.unitm if raw and s.unitm else "")

# --- modes ----------------------------------------------------------------
def dump_once(fd):
    refused = []
    def run(label, name, fn):
        try:
            print("  %-22s %s" % (label + ":", fn()))
        except QueryError as e:
            refused.append(str(e)); print("  %-22s <refused: %s>" % (label + ":", e.args[0].split(" - ")[0]))

    run("AIE version", "AIE_VERSION",
        lambda: "%d.%d" % ((v := query_struct(fd, "AIE_VERSION", AieVersion)).major, v.minor))
    run("AIE metadata", "AIE_METADATA", lambda: (lambda m: "%d cols x %d rows, col_size %d B; "
        "core rows %d@%d, mem rows %d@%d, shim rows %d@%d" % (m.cols, m.rows, m.col_size,
        m.core.row_count, m.core.row_start, m.mem.row_count, m.mem.row_start,
        m.shim.row_count, m.shim.row_start))(query_struct(fd, "AIE_METADATA", AieMetadata)))
    run("Firmware version", "FIRMWARE_VERSION", lambda: (lambda f: "%d.%d.%d build %d" %
        (f.major, f.minor, f.patch, f.build))(query_struct(fd, "FIRMWARE_VERSION", FwVersion)))
    run("Clocks", "CLOCK_METADATA", lambda: (lambda c: "%s = %d MHz, %s = %d MHz" %
        (c.mp_npu_clock.name.decode(), c.mp_npu_clock.freq_mhz, c.h_clock.name.decode(),
         c.h_clock.freq_mhz))(query_struct(fd, "CLOCK_METADATA", ClockMetadata)))
    run("Power mode", "POWER_MODE", lambda: (lambda p: "%s (%d)" % (POWER_MODES[p.power_mode]
        if p.power_mode < len(POWER_MODES) else "?", p.power_mode))(query_struct(fd, "POWER_MODE", PowerMode)))
    run("Resource info", "RESOURCE_INFO", lambda: (lambda r: "H clk max %d MHz; TOPS max %d curr %d; "
        "hwctx max %d curr %d" % (r.npu_clk_max, r.npu_tops_max, r.npu_tops_curr, r.npu_task_max,
        r.npu_task_curr))(query_struct(fd, "RESOURCE_INFO", ResourceInfo)))
    run("Force preempt", "FORCE_PREEMPT_STATE",
        lambda: "on" if query_struct(fd, "FORCE_PREEMPT_STATE", AttrState).state else "off")
    run("Frame-boundary preempt", "FRAME_BOUNDARY_PREEMPT_STATE",
        lambda: "on" if query_struct(fd, "FRAME_BOUNDARY_PREEMPT_STATE", AttrState).state else "off")
    run("Sensors", "SENSORS", lambda: (lambda ss: "%d: " % len(ss) + ", ".join(
        "%s [%s]" % (sensor_str(s, raw=True), SENSOR_TYPES.get(s.type, s.type)) for s in ss)
        if ss else "0 (driver returned no sensor entries)")(sensors(fd)))
    def hwctx():
        hs = hwctxs(fd)
        root_note = "" if os.geteuid() == 0 else "  (non-root: driver may hide contexts of other uids)"
        if not hs: return "0" + root_note
        return "%d%s" % (len(hs), root_note) + "".join(
            "\n    ctx %-3d pid %-7d cols %d+%d  submit %d done %d migr %d preempt %d err %d" %
            (h.context_id, h.pid, h.start_col, h.num_col, h.command_submissions,
             h.command_completions, h.migrations, h.preemptions, h.errors) for h in hs)
    run("HW contexts", "HW_CONTEXTS", hwctx)
    print("  %-22s not queried (needs a firmware mailbox round-trip with a typed header)" % "Telemetry:")
    if refused:
        print("Refused queries:"); [print("  - " + r) for r in refused]
    return refused

def watch(fd, interval, count):
    print("%-12s %8s %8s %-8s %8s %6s %5s  %s" %
          ("time", "MP-NPU", "H-clk", "pwrmode", "TOPS", "hwctx", "ctx", "sensors"))
    dead = set(); n = 0
    def q(name, fn):
        if name in dead: return None
        try: return fn()
        except QueryError as e:
            dead.add(name); print("  ! %s" % e); return None
    while count == 0 or n < count:
        t = time.time(); ts = time.strftime("%H:%M:%S", time.localtime(t)) + ".%03d" % int(t * 1000 % 1000)
        c = q("CLOCK_METADATA", lambda: query_struct(fd, "CLOCK_METADATA", ClockMetadata))
        p = q("POWER_MODE", lambda: query_struct(fd, "POWER_MODE", PowerMode))
        r = q("RESOURCE_INFO", lambda: query_struct(fd, "RESOURCE_INFO", ResourceInfo))
        hs = q("HW_CONTEXTS", lambda: hwctxs(fd))
        ss = q("SENSORS", lambda: sensors(fd))
        print("%-12s %8s %8s %-8s %8s %6s %5s  %s" % (
            ts, "%d MHz" % c.mp_npu_clock.freq_mhz if c else "-", "%d MHz" % c.h_clock.freq_mhz if c else "-",
            POWER_MODES[p.power_mode] if p and p.power_mode < len(POWER_MODES) else "-",
            "%d/%d" % (r.npu_tops_curr, r.npu_tops_max) if r else "-",
            "%d/%d" % (r.npu_task_curr, r.npu_task_max) if r else "-",
            str(len(hs)) if hs is not None else "-",
            " ".join(sensor_str(s) for s in ss if s.type != 1) if ss else "-"), flush=True)
        n += 1
        if count == 0 or n < count: time.sleep(interval)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default=DEV)
    ap.add_argument("--watch", type=float, metavar="SEC", help="poll every SEC seconds")
    ap.add_argument("--count", type=int, default=0, help="stop after N samples (0 = forever)")
    a = ap.parse_args()
    try:
        fd = os.open(a.dev, os.O_RDWR | os.O_CLOEXEC)
    except OSError as e:
        sys.exit("cannot open %s: %s%s" % (a.dev, e.strerror,
                 " (need to be in the 'render' group or root)" if e.errno in (errno.EACCES, errno.EPERM) else ""))
    try:
        if a.watch:
            watch(fd, a.watch, a.count)
        else:
            print("amdxdna GET_INFO on %s  (ioctl 0x%X, uid %d%s)" % (a.dev, IOCTL_GET_INFO,
                  os.geteuid(), "" if os.geteuid() == 0 else ", not root"))
            dump_once(fd)
    except KeyboardInterrupt:
        pass
    finally:
        os.close(fd)

if __name__ == "__main__":
    main()
