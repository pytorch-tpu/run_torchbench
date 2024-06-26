import torch
import time
import torch_xla2
import torch_xla2.interop
import os
import importlib 
import sys
import logging
import sys

# NOTE: replace this patch below with your installation
TORCH_BENCH_PATH = 'benchmark'
sys.path.append(TORCH_BENCH_PATH)

model_name = "torchbenchmark.models.detectron2_fasterrcnn_r_50_c4" # replace this by the name of the model you're working on
module = importlib.import_module(model_name)
benchmark_cls = getattr(module, "Model", None)
benchmark = benchmark_cls(test="eval", device = "cpu") # test = train or eval device = cuda or cpu

model, example = benchmark.get_module()

env = torch_xla2.default_env()
env.config.debug_print_each_op = False
model = env.to_xla(model)
example = env.to_xla(example)
with env:
    start = time.perf_counter()
    print(model(*example))
    end = time.perf_counter()
    print('Eager mode time', end - start)


def func_call(state, example):
    return torch.func.functional_call(model, state, example, tie_weights=False)

jitted = torch_xla2.interop.jax_jit(func_call)
start = time.perf_counter()
print(func_call(model.state_dict(), example))
end = time.perf_counter()
print('Jitted mode time', end - start)

