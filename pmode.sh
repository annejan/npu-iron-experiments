#!/bin/bash
# Set the NPU power mode (needs root: DRM_IOCTL_AMDXDNA_SET_STATE). sudo drops LD_LIBRARY_PATH, so pass it.
#   ./pmode.sh performance      1.81 GHz vs 1.40 GHz default on a Strix Point under Windows (ghost-intuit write-up);
#                               no clear effect measured on this Linux box, see README
#   ./pmode.sh default
RAI=$(cd "$(dirname "$0")/../ryzenai" && pwd)
LIBS=$RAI/xrt/lib:$RAI/boost/usr/lib/x86_64-linux-gnu
sudo env LD_LIBRARY_PATH="$LIBS" XILINX_XRT="$RAI/xrt" "$RAI/xrt/bin/unwrapped/xrt-smi" configure --pmode "${1:-performance}"
env LD_LIBRARY_PATH="$LIBS" XILINX_XRT="$RAI/xrt" "$RAI/xrt/bin/unwrapped/xrt-smi" examine -r platform | grep -i 'power mode'
