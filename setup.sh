#!/bin/bash
# IRON (mlir-aie v1.4.5) on Tumbleweed, with the unpacked XRT 2.26 from ../ryzenai (no system install).
#   ./setup.sh          one time: venv + wheels
#   ./env.sh CMD...     every time: run CMD with XRT + IRON in the environment
set -euo pipefail
cd "$(dirname "$0")"
python3.12 -m venv ironenv
ironenv/bin/pip install -q --upgrade pip
ironenv/bin/pip install mlir_aie==1.4.5 -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.5
ironenv/bin/pip install -r mlir-aie/utils/peano-requirements.txt
ironenv/bin/pip install -r mlir-aie/python/requirements.txt
du -sh ironenv
ironenv/bin/pip list 2>/dev/null | grep -i -E "mlir|llvm-aie|numpy"
