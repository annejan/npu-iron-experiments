#!/bin/bash
# IRON (mlir-aie v1.4.5) on openSUSE Tumbleweed, with XRT 2.26 unpacked in ../ryzenai (no system install;
# see README "Setup" for the .debs and the kernel driver).
#   ./setup.sh          one time: mlir-aie checkout + venv + wheels (~2 GB)
#   ./env.sh CMD...     every time: run CMD with XRT + IRON in the environment
set -euo pipefail
cd "$(dirname "$0")"
[ -d mlir-aie ] || git clone --depth 1 --branch v1.4.5 https://github.com/Xilinx/mlir-aie.git   # no submodules needed
# 3.12, not newer: ../ryzenai/xrt/python/pyxrt is a CPython 3.12 extension and the mlir_aie wheel is cp312
# (Tumbleweed: zypper in python312)
python3.12 -m venv ironenv
ironenv/bin/pip install -q --upgrade pip
ironenv/bin/pip install mlir_aie==1.4.5 -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.5
ironenv/bin/pip install -r mlir-aie/utils/peano-requirements.txt
ironenv/bin/pip install -r mlir-aie/python/requirements.txt
# what the demos and probes import on top of IRON: windows, PNG/webcam, wav/live audio, static schedule reports
ironenv/bin/pip install -q pygame-ce opencv-python soundfile sounddevice scipy pyyaml
du -sh ironenv
ironenv/bin/pip list 2>/dev/null | grep -i -E "mlir|llvm-aie|numpy"
