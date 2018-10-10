export NUMBAPRO_NVVM=/opt/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/opt/cuda/nvvm/libdevice

BIN_FILE=$(readlink -f $0)
PROJ_HOME=$(dirname $BIN_FILE)
export PYTHONPATH=$PROJ_HOME
