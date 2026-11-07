#!/bin/bash
# Set the NPU power mode (needs root: DRM_IOCTL_AMDXDNA_SET_STATE). sudo drops LD_LIBRARY_PATH, so pass it.
#   ./pmode.sh performance      1.81 GHz on Strix (paper), vs 1.40 GHz default
#   ./pmode.sh default
RAI=$(cd "$(dirname "$0")/../ryzenai" && pwd)
LIBS=$RAI/xrt/lib:$RAI/boost/usr/lib/x86_64-linux-gnu
sudo env LD_LIBRARY_PATH="$LIBS" XILINX_XRT="$RAI/xrt" "$RAI/xrt/bin/unwrapped/xrt-smi" configure --pmode "${1:-performance}"
env LD_LIBRARY_PATH="$LIBS" XILINX_XRT="$RAI/xrt" "$RAI/xrt/bin/unwrapped/xrt-smi" examine -r platform | grep -i 'power mode'
