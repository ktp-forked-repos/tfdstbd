# https://www.tensorflow.org/extend/adding_an_op#compile_the_op_using_your_system_compiler_tensorflow_binary_installation

TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
ICU_FLAGS := $(shell pkg-config --libs --cflags icu-uc)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	OS_FLAGS := -undefined dynamic_lookup
endif

all:
	g++ -std=c++11 -shared sentsegm/ops/ops.cc -o sentsegm/ops/ops.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2 $(OS_FLAGS) $(ICU_FLAGS)
	hash python 2>/dev/null && echo "Test with default Python" && python -m sentsegm.test.ops
	hash python2 2>/dev/null && echo "Test with Python v2" && python2 -m sentsegm.test.ops
	hash python3 2>/dev/null && echo "Test with Python v3" && python3 -m sentsegm.test.ops



