# Anomaly Detection & MLLM Fine-Tuning

This directory contains the code for the final anomaly detection stage of the pipeline. It leverages a fine-tuned Multimodal Large Language Model (MLLM) to perform highly accurate, context-aware industrial anomaly diagnosis.

Since we did not make substantial modifications to the core training loop of the official MLLM fine-tuning framework, **this directory only provides the code for inference, post-processing, and anomaly prompt generation**. 

## 🛠️ Environment Setup & Fine-Tuning Guide

For model initialization, environment configuration, and the actual fine-tuning execution, please refer directly to the official documentation of the base model (e.g., [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)).

> ⚠️ **CRITICAL: Fine-Tuning Task Selection**
> When configuring the fine-tuning task according to the official repository's documentation, **ensure you select the "Image Captioning / Image Description" format**. 
> DO NOT use the "Visual Grounding / Bounding Box Localization" format. Our method trains the MLLM to output rich, semantic diagnostic narratives rather than coordinates.

## 📂 Directory Contents & Pipeline Steps

### 1. Prompt Generation 
Before fine-tuning, you must prepare the training corpus. The scripts in this module combine weak anomaly labels with cross-modal multi-domain clustering results to generate information-dense image-text pairs.

### 2. MLLM Fine-Tuning 
Use the combined dataset to fine-tune the MLLM using the official framework's instructions.

### 3. Inference 
Once the model is fine-tuned, use this script to run anomaly detection on the test set. It loads the fine-tuned weights and processes TEDS images to output diagnostic descriptions.

