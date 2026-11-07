#!/bin/bash
# Run a command with XRT 2.26 (unpacked in ../ryzenai, see README "Setup") and IRON (ironenv) set up.
#   ./env.sh python fun/mandel.py                    any script (first call of a design compiles: 0.4-1.7 s)
#   ./env.sh bash                                    a shell to play in
#   ./env.sh make -C mlir-aie/programming_examples/basic/vector_scalar_mul run_py
HERE=$(cd "$(dirname "$0")" && pwd)
RAI=$HERE/../ryzenai
export XILINX_XRT=$RAI/xrt
export PATH=$XILINX_XRT/bin:$PATH
export LD_LIBRARY_PATH=$XILINX_XRT/lib:$RAI/boost/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$XILINX_XRT/python${PYTHONPATH:+:$PYTHONPATH}
source "$HERE/ironenv/bin/activate"
ARGS=("$@"); set --            # env_setup.sh is sourced and would read our argv as its own
cd "$HERE/mlir-aie" && source utils/env_setup.sh >/dev/null && cd - >/dev/null
exec "${ARGS[@]}"
