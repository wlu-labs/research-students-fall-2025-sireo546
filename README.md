[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20322509&assignment_repo_type=AssignmentRepo)
Project Instructions
==============================

The goal is to compare several quantization methods on a small neural network:

- **FP32 baseline**
- **PTQ** – Post Training Quantization  
- **QAT** – Quantization Aware Training  
- **GPTQ (simulated)** – Gradient-based PTQ (size/latency simulated in this script)

All final experiments were run on **Rorqual (CPU node)** using **Python 3.10** and **PyTorch 2.6.0** with the `fbgemm` quantization backend.

---

## Repository Structure

```text
.
├── src/
│   ├── data/
│   │   └── trafficdata.csv                # Tabular traffic dataset used in all experiments
│   │
│   ├── models/                          # Original QAT demo code from the project
    │   ├── diagram_qat.py
│   │   ├── predict_model.py
│   │   ├── qat_demo.py
│   │   ├── qat_results.txt  
│   │   └── train_model.py
│   │
│   ├── qat_comparisons/                   # **Older local experiments (Beluga / Mac)**
│   │   ├── base_fp32.pt
        ├── compare_script.py
│   │   ├── ptq_model.pt
│   │   ├── qat_model.pt
│   │   ├── trafficdata.csv
│   │   └── initial / updated result logs
│   │
│   ├── qat_comparisons_beluga/            # **Archived experiments from Beluga**
│   │   ├── best results so far.txt        #latest result
│   │   ├── compare_script.py
│   │   ├── results                        #previous results
│   │   └── results beluga.jpg             #previous results
│   │
│   └── quantization_FINAL_rorqual/        # ✅ **Final code + results (Rorqual)**
│     
        ├── base_fp32.pt                   # Saved FP32 model
│       ├── compare_script.py              # Core implementation (PTQ, QAT, GPTQ)
        ├── final_results_rorqual.txt      # Console output from the best Rorqual run
│       ├── ptq_model.pt                   # Saved PTQ model
│       ├── qat_model.pt                   # Saved QAT model
│       └── trafficdata.csv                # Copy of the dataset used by the script
├── __init__.py
├── main.py                                # Entry point – runs the full comparison
├── literature.md                          # Notes / related work summary
└── requirements.txt                       # Python dependencies
