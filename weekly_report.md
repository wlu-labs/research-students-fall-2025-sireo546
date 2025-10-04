## Weekly Report
- This file contains a paragraph of 1000 Characters or more about the progress made by the student for each week. I am creating the place holder for it.

## Week 1 (Date: Sept 8–14, 2025)

This week I focused on federated learning and read five papers that showed its uses and challenges. *A Survey of Federated Learning for Connected and Automated Vehicles* explained how FedAvg lets cars train models locally, which helps protect privacy and cut network costs, but it still struggles with non-IID data, forgetting past knowledge, and limited testing in real settings. *Federated Continual Learning* looked at how devices can keep learning over time while avoiding forgetting, using methods like replay and regularization, but it also faces problems with communication costs and unstable training. *Federated Learning with Digital Twins for Trajectory Prediction* introduced FedSTAST, which used attention to improve predictions, though it was only tested in simulations. *Towards Explainable Traffic Flow Prediction with LLMs* treated traffic as a language problem and showed how LLaMA 2 with LoRA could give both predictions and explanations, but required high compute. Finally, *Evaluating Quantized LLMs for Code Generation* tested code models at different bit levels, showing that 4-bit worked well on laptops but reduced accuracy. Overall, these papers showed how FL and quantization both try to balance accuracy, efficiency, and privacy.

---

## Week 2 (Date: Sept 15–21, 2025)

This week I focused on quantization and the basics of large language models. The *LLM-QAT: Data-Free Quantization Aware Training* paper explained how models can be compressed by lowering the precision of weights, activations, and KV-caches, while still keeping good performance. It used data-free knowledge distillation, where a teacher model makes synthetic data, and quantization-aware training, where the student model learns to work with low-bit numbers. I also watched a simple video that explained PTQ (post-training quantization), QAT, and dynamic quantization, showing how they make models faster, cheaper, and easier to run without powerful GPUs. To build a stronger foundation, I started Chapter 1 of *Build a Large Language Model from Scratch*. It covered what LLMs are, why transformers are important, and how pretraining and finetuning make models generalize to many tasks. Taken together, the paper, video, and book gave me both a practical and theoretical view: I saw how quantization makes models efficient and how LLMs are designed to work in the first place.


### Week 3 (Date: Sept 22–28, 2025))
This week I ran my first QAT demo (qat_demo.py) in PyTorch. I trained a small FP32 model with fake quantization layers and converted it to INT8, confirming the Linear layers turned into QuantizedLinear. I also started testing a quick model size comparison (FP32 vs INT8), uploaded my code to the GitHub repo, and continued Flower AI tutorials to prepare for federated learning.


### Week 4 Date: Sept 29–Oct 5, 2025)

This week I focused on interpreting the QAT demo outputs, especially scale/zero_point values and the training loss. I worked on debugging the FP32 vs INT8 size comparison and saved the run logs to qat_results.txt in my repo. I also registered for the CCDB. Worked on making a flow chart to keep my accountable.

### Week 5 (Date:   )


### Week 6 (Date:   )
