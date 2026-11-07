import sys; sys.path.insert(0, "fun")
from pathlib import Path
import numpy as np
from aie.utils.compile import remarks
from aie.iron import ExternalFunction
import grayscott as G
band_ty = np.ndarray[(G.ROWS * 2 * G.S,), np.dtype[np.int16]]; edge_ty = np.ndarray[(2 * G.S,), np.dtype[np.int16]]
d = Path(".remarks/rem_gs"); d.mkdir(parents=True, exist_ok=True)
for name, src, args in (("gs_step", G.kstep(0.037, 0.06), [band_ty, band_ty, edge_ty, edge_ty]), ("gs_row", G.krow(), [band_ty, edge_ty, np.dtype(np.int32)])):
    rep, why = remarks.analyze(ExternalFunction(name, source_string=src, arg_types=args), "aie2p", d)
    if rep is None: print(name, why[:1500]); continue
    print(name, "libcalls", rep.libcalls, "stack", rep.kernel_stack_bytes, "notes", rep.schedule_notes)
    for key, l in rep.loops.items(): print("  loop", key, "ii", l.ii, "ns", l.ns, "zol", l.zol, "trips", l.trips, "bundles", l.bundle_count)
