# A Semi-Supervised Multi-Stage Pipeline for Low-Resource Industrial Condition Monitoring

This repository contains the official implementation code for the paper: **"A Semi-Supervised Multi-Stage Pipeline for Low-Resource Industrial Condition Monitoring by Injecting Domain Knowledge via Cross-Modal MLLM Fine-Tuning"**. The source code of this study
will be publicly available upon publication of paper.

## ⚙️ Hyperparameters and Configuration

We largely followed the official hyperparameter settings provided in the [BEiT-v2](https://github.com/microsoft/unilm/tree/7ae2ee53bf7fff85e730c72083b7e999b0b9ba44/beit2) and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune) repositories.

> **⚠️ Note on Learning Rate:**
> The learning rate was scaled linearly based on the effective batch size adapted for our hardware setup (number of GPUs and VRAM), following the standard linear scaling rule:
>
> $$\text{New LR} = \text{Base LR} \times \frac{\text{New Batch Size}}{\text{Base Batch Size}}$$

## 🚀 Usage Pipeline

The code involves multiple stages. Please follow the order below to reproduce the pipeline described in the paper:

### Stage 1: Unit Segmentation
1.  **Train the Model:**
    Navigate to the **`Unit segmentation/`** folder to train the visual model.
    * *Logs:* Training logs and running records for this stage can be found in the **`Running record/`** folder.

2.  **Inference with Sliding Window:**
    Use the trained model to process EMU images. The inference logic, including the adaptive window mechanism, is located in the **`Multi scale sliding window/`** folder.

3.  **Result Filtering:**
    Refine the segmentation results by removing physically implausible predictions using the engineering rules defined in the **`Knowledge based filtering/`** folder.

### Stage 2: Anomaly Detection & MLLM Fine-Tuning
4.  **Clustering & Dataset Construction:**
    Generate pseudo-labels for anomaly images (via clustering) and organize the recognition results into image-text pairs. Code for these preprocessing steps is located in the **`Anomaly detection/`** folder.

5.  **Fine-Tune MLLM:**
    Fine-tune the Multimodal Large Language Model (MLLM) using the generated dataset.
    * *Source Code:* Refer to the **`Anomaly detection/`** folder.
    * *Logs:* Fine-tuning records are available in the **`Running record/`** folder.

## 🔗 Related Resources / Acknowledgements

Our work builds upon and uses code from the following excellent open-source projects:

* **[BEiT-v2](https://github.com/microsoft/unilm/tree/7ae2ee53bf7fff85e730c72083b7e999b0b9ba44/beit2)**: We used their implementation as the foundation for our unit segmentation model (Stage 1).
* **[Open-clip](https://github.com/mlfoundations/open_clip)**: We used their implementation as the teacher model for our unit segmentation model (Stage 1).
* **[DINOv2](https://github.com/facebookresearch/dinov2/tree/main/dinov2)**: We used their implementation as the teacher model for our unit segmentation model (Stage 1).
* **[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune)**: Our anomaly detection MLLM is based on their fine-tuning code (Stage 2).

