import torch
import time
import torch_xla2
import torch_xla2.interop
import os
import importlib 
import sys
import logging
import sys


def main():
    # NOTE: replace this patch below with your installation
    TORCH_BENCH_PATH = 'benchmark'
    sys.path.append(TORCH_BENCH_PATH)

    # Get model from torch bench
    model_name = "torchbenchmark.models.hf_clip"
    module = importlib.import_module(model_name)
    benchmark_cls = getattr(module, "Model", None)
    benchmark = benchmark_cls(test="eval", device = "cpu")
    model, example = benchmark.get_module()

    # Run the model once in torch
    expected = model(**example)


    env = torch_xla2.default_env()
    # Debug options to change:
    env.config.debug_print_each_op = False
    env.config.debug_accuracy_for_each_op = False
    env.config.debug_mixed_tensor = False

    # Move the model to "xla" i.e. jax
    model = env.to_xla(model)
    example = env.to_xla(example)
    with env:
        start = time.perf_counter()
        xla2_ans = model(**example)
        print(example)
        end = time.perf_counter()
        print('Eager mode time', end - start)
    print('Eager max abs vs expected: logits_per_image', (torch_xla2.tensor.j2t(xla2_ans.logits_per_image._elem) - expected.logits_per_image).abs().max())
    print('Eager max abs vs expected: logits_per_text', (torch_xla2.tensor.j2t(xla2_ans.logits_per_text._elem) - expected.logits_per_text).abs().max())
    print('Eager max abs vs expected: text_embeds', (torch_xla2.tensor.j2t(xla2_ans.text_embeds._elem) - expected.text_embeds).abs().max())
    print('Eager max abs vs expected: image_embeds', (torch_xla2.tensor.j2t(xla2_ans.image_embeds._elem) - expected.image_embeds).abs().max())
    print('Eager max abs vs expected: text_model_output', (torch_xla2.tensor.j2t(xla2_ans.text_model_output._elem) - expected.text_model_output).abs().max())
    print('Eager max abs vs expected: vision_model_output', (torch_xla2.tensor.j2t(xla2_ans.vision_model_output._elem) - expected.vision_model_output).abs().max())

    def func_call(state, example):
      with env:
        return torch.func.functional_call(model, state, None, example, tie_weights=False)

    # doing it jitted
    jitted = torch_xla2.interop.jax_jit(func_call)
    start = time.perf_counter()
    xla2_ans = func_call(model.state_dict(), example)
    end = time.perf_counter()
    print('Jitted mode time', end - start)
    print('Jitted max abs vs expected: logits_per_image', (torch_xla2.tensor.j2t(xla2_ans.logits_per_image._elem) - expected.logits_per_image).abs().max())
    print('Jitted max abs vs expected: logits_per_text', (torch_xla2.tensor.j2t(xla2_ans.logits_per_text._elem) - expected.logits_per_text).abs().max())
    print('Jitted max abs vs expected: text_embeds', (torch_xla2.tensor.j2t(xla2_ans.text_embeds._elem) - expected.text_embeds).abs().max())
    print('Jitted max abs vs expected: image_embeds', (torch_xla2.tensor.j2t(xla2_ans.image_embeds._elem) - expected.image_embeds).abs().max())
    print('Jitted max abs vs expected: text_model_output', (torch_xla2.tensor.j2t(xla2_ans.text_model_output._elem) - expected.text_model_output).abs().max())
    print('Jitted max abs vs expected: vision_model_output', (torch_xla2.tensor.j2t(xla2_ans.vision_model_output._elem) - expected.vision_model_output).abs().max())
    return 0


if __name__ == '__main__':
    sys.exit(main())

