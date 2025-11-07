# Models

- **qat_demo.py** — small PyTorch Quantization Aware Training demo  
  (FP32 → fake quant → train → convert to INT8)
- **diagram_qat.py** — simple plot that shows the QAT workflow

## Notes from QAT Demo
- An **epoch** = one full pass through training data
- **Loss** = measure of error (lower is better)
- `convert()` = swaps fake quant layers for real int8 ops
- `scale/zero_point` = maps FP32 numbers into int8 range
