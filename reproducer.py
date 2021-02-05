import pyiree as iree
import pyiree.compiler2
import pyiree.rt

import numpy as np
from numpy import dtype

binary = iree.compiler2.compile_file("./mobilenet_v3_reproducer.mlir",
                                     target_backends=["dylib-llvm-aot"],
                                     extra_args=["-iree-llvm-sanitize=address"])
cpp_vm_module = iree.rt.VmModule.from_flatbuffer(binary)
module = iree.rt.load_module(cpp_vm_module, config=iree.rt.Config("dylib"))

signature = [
    ((), dtype('int32')),
    ((1, 1, 16, 16), dtype('float32')),
    ((1, 1, 16, 16), dtype('float32')),
    ((3, 3, 1, 16), dtype('float32')),
    ((8,), dtype('float32')),
    ((1, 1, 16, 8), dtype('float32')),
    ((16,), dtype('float32')),
    ((1, 1, 8, 16), dtype('float32')),
    ((1, 16, 16, 16), dtype('float32')),
    ((1,), dtype('int32')),
]
ndarange = lambda shape, dtype: np.arange(np.prod(shape), dtype=dtype).reshape(shape)
inputs = [ndarange(*args) for args in signature]

baseline = module.main(*inputs)
for i in range(1000):
  outputs = module.main(*inputs)
  for a, b in zip(baseline, outputs):
    np.testing.assert_equal(a, b)  # just doing something similar to what the original script does.
