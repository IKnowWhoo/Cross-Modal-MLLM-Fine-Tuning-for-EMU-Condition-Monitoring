# A Semi-Supervised Multi-Stage Pipeline for Low-Resource Industrial Condition Monitoring

This repository contains the official implementation code for the paper: **"A Semi-Supervised Multi-Stage Pipeline for Low-Resource Industrial Condition Monitoring by Injecting Domain Knowledge via Cross-Modal MLLM Fine-Tuning"**.

## ⚙️ Hyperparameters and Configuration

We largely followed the official hyperparameter settings provided in the [BEiT-v2](https://github.com/microsoft/unilm/tree/7ae2ee53bf7fff85e730c72083b7e999b0b9ba44/beit2) and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune) repositories.

> **⚠️ Note on Learning Rate:**
> The learning rate was scaled linearly based on the effective batch size adapted for our hardware setup (number of GPUs and VRAM), following the standard linear scaling rule:
>
> $$\text{New LR} = \text{Base LR} \times \frac{\text{New Batch Size}}{\text{Base Batch Size}}$$

## 🚀 Usage Pipeline

Please run the code in the following sequential order to reproduce the pipeline described in the paper:

1.  **Train Unit Segmentation Model:**
    Train the visual model using the code in the `Unit segmentation/` directory.
    * *Reference:* We have provided our training logs/running records for comparison.

2.  **Inference & Filtering:**
    Process the EMU images using the trained segmentation model combined with the **multi-scale sliding window** mechanism.
    * Apply the **knowledge-based filter** to refine the segmentation results and remove physically implausible predictions.

3.  **Anomaly Clustering:**
    Perform clustering on the anomaly images to generate pseudo-labels (addressing the long-tail distribution issue).

4.  **Dataset Construction:**
    Organize the recognition results, sensor data, and pseudo-labels to construct the detailed **image-text pair dataset**.

5.  **Fine-tune MLLM:**
    Fine-tune the Multimodal Large Language Model (MLLM) using the generated cross-modal image-text pairs.

## 🔗 Related Resources / Acknowledgements

Our work builds upon and uses code from the following excellent open-source projects:

* **[BEiT-v2](https://github.com/microsoft/unilm/tree/7ae2ee53bf7fff85e730c72083b7e999b0b9ba44/beit2)**: We used their implementation as the foundation for our unit segmentation model (Stage 1).
* **[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune)**: Our anomaly detection MLLM is based on their fine-tuning code (Stage 3).
