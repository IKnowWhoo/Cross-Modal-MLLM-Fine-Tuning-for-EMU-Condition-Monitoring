# Cross-Modal-MLLM-Fine-Tuning-for-EMU-Condition-Monitoring
Code for the paper "A Semi-Supervised Multi-Stage Pipeline for Low-Resource Industrial Condition Monitoring by Injecting Domain Knowledge via Cross-Modal MLLM Fine-Tuning"

# Hyperparameters and Configuration

We largely followed the official hyperparameter settings provided in the BEiT-v2 and Qwen2.5-VL repositories.

Note on Learning Rate: The learning rate was scaled linearly based on the effective batch size adapted for our hardware setup (number of GPUs and VRAM), following the linear scaling rule: New LR = Base LR × (New Batch Size / Base Batch Size)
