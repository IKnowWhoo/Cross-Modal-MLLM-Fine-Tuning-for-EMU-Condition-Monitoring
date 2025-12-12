# Cross-Modal-MLLM-Fine-Tuning-for-EMU-Condition-Monitoring
Code for the paper "A Semi-Supervised Multi-Stage Pipeline for Low-Resource Industrial Condition Monitoring by Injecting Domain Knowledge via Cross-Modal MLLM Fine-Tuning"

# Hyperparameters and Configuration

We largely followed the official hyperparameter settings provided in the **BEiT-v2** and **Qwen2.5-VL** repositories.

Note on Learning Rate: The learning rate was scaled linearly based on the effective batch size adapted for our hardware setup (number of GPUs and VRAM), following the linear scaling rule: New LR = Base LR × (New Batch Size / Base Batch Size)

## 🔗 Related Resources

Our work builds upon and uses code from the following excellent open-source projects:
* **[BEiT-v2](https://github.com/microsoft/unilm/tree/7ae2ee53bf7fff85e730c72083b7e999b0b9ba44/beit2)**: We used their implementation as the foundation for our unit segmentation model.
* **[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune)**: Our multi-modal large language model (MLLM) head is based on their fine-tuning code.
