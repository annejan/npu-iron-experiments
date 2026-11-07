import sys; sys.path.insert(0, "fun")
from pathlib import Path
import numpy as np
from aie.utils.compile import remarks
from aie.iron import ExternalFunction
import ising
band_ty = np.ndarray[(ising.ROWS * ising.S,), np.dtype[np.int16]]; edge_ty = np.ndarray[(ising.S,), np.dtype[np.int16]]
d = Path(__file__).resolve().parent.parent / ".remarks" / ("rem_ising"); d.mkdir(parents=True, exist_ok=True)
for name, src, args in (("ising_step", ising.kstep(2.269), [band_ty, band_ty, edge_ty, edge_ty, np.dtype(np.int32), np.dtype(np.int32)]),
                        ("ising_row", ising.krow(), [band_ty, edge_ty, np.dtype(np.int32)])):
    rep, why = remarks.analyze(ExternalFunction(name, source_string=src, arg_types=args), "aie2p", d)
    if rep is None: print(name, why[:2500]); continue
    print(name, "libcalls", rep.libcalls, "stack", rep.kernel_stack_bytes, "notes", rep.schedule_notes)
    for key, l in rep.loops.items(): print("  loop", key, "ii", l.ii, "ns", l.ns, "zol", l.zol, "trips", l.trips, "bundles", l.bundle_count)
