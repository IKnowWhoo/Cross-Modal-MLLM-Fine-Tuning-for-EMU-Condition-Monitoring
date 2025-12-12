Unit Segmentation Implementation
# Overview
This repository contains the implementation for unit segmentation based on BEiTv2 architecture. The code includes additional visualization capabilities compared to the original implementation and supports using both Open-CLIP and DINOv2 as teacher models for enhanced performance.

Prerequisites
1. BEiTv2 Installation
All unit segmentation implementations are built upon BEiTv2. You must install BEiTv2 first:

Clone and install BEiTv2
git clone https://github.com/microsoft/unilm.git
cd unilm/beit2
pip install -r requirements.txt
pip install -e .

2. Teacher Model Dependencies (Optional)
If you plan to use Open-CLIP and/or DINOv2 as teacher models:

Open-CLIP Installation:
bash

DINOv2 Installation:
bash
