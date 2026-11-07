import sys; sys.path.insert(0, "fun")
from pathlib import Path
from aie.utils.compile import remarks
import numpy as np
import reverb
from aie.iron import ExternalFunction
obj_ty = np.ndarray[(reverb.OBJ,), np.dtype[np.int16]]; part_ty = np.ndarray[(reverb.B,), np.dtype[np.int32]]
src = reverb.SOURCES[sys.argv[1]] if len(sys.argv) > 1 else reverb.KSRC
ef = ExternalFunction("fir", source_string=src, arg_types=[obj_ty, part_ty])
d = Path(".remarks/rem_fir"); d.mkdir(parents=True, exist_ok=True)
rep, why = remarks.analyze(ef, "aie2p", d)
if rep is None: print(why); sys.exit(1)
print("libcalls", rep.libcalls, "stack", rep.kernel_stack_bytes, "notes", rep.schedule_notes)
for key, l in rep.loops.items(): print("  loop", key, "ii", l.ii, "ns", l.ns, "zol", l.zol, "trips", l.trips, "bundles", l.bundle_count)
