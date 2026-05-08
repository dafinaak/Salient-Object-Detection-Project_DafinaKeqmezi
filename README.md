# Salient Object Detection from Scratch (ECSSD)

A CNN-based Salient Object Detection (SOD) system implemented entirely from scratch in PyTorch. Given any input image, the model produces a binary mask that highlights the most visually dominant object.

This project was completed as an end-to-end ML/DL exercise: data pipeline → model design → training loop → evaluation → experiments → live demo.

---

## Table of Contents

- [Results](#results)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Demo](#demo)
- [Method](#method)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Author](#author)
- [License](#license)

---

## Results

Tested on the held-out 15% split of ECSSD (150 images). The improved model (BatchNorm + U-Net skip connections) substantially outperforms the baseline across every metric:

| Metric    | Baseline | Improved |       Δ       |
|-----------|---------:|---------:|--------------:|
| IoU       |   0.4294 |   0.5777 | **+34.5 %**   |
| Precision |   0.5951 |   0.6778 |     +13.9 %   |
| Recall    |   0.6458 |   0.8137 |     +26.0 %   |
| F1        |   0.5728 |   0.7076 |     +23.5 %   |
| MAE ↓     |   0.2423 |   0.1650 |     −31.9 %   |

Inference runs at ~21 ms per 128×128 image on a T4 GPU.

---

## Repository Structure

```
Salient-Object-Detection-Project_DafinaKeqmezi/
├── data_loader.py          
├── sod_model.py           
├── train.py                
├── evaluate.py             
├── app.py                  
├── demo_notebook.ipynb    
├── requirements.txt
├── README.md
├── report_assets/          
└── checkpoints/            
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+ and torchvision 0.15+
- OpenCV, NumPy, Matplotlib, scikit-learn, tqdm
- Gradio 4.0+ (for the web demo)

A CUDA-capable GPU is recommended for training; inference runs comfortably on CPU.

See [requirements.txt](requirements.txt) for the full pinned list.

---

## Setup

```bash
git clone https://github.com/dafinaak/Salient-Object-Detection-Project_DafinaKeqmezi
cd Salient-Object-Detection-Project_DafinaKeqmezi
pip install -r requirements.txt
```

### Dataset

Download the ECSSD dataset (1000 image + mask pairs). The expected layout is:

```
data/ECSSD/
├── images/      # 0001.jpg ... 1000.jpg
└── masks/       # 0001.png ... 1000.png
```

Adjust the `--img_dir` and `--mask_dir` flags below to point at your folders.

---

## Training

Baseline model:

```bash
python train.py --img_dir data/ECSSD/images --mask_dir data/ECSSD/masks
```

Improved model (BatchNorm + skip connections):

```bash
python train.py --img_dir data/ECSSD/images --mask_dir data/ECSSD/masks --improved
```

Checkpoints are saved every epoch to `checkpoints/`. If training is interrupted, just run the same command again — it will automatically resume from the last checkpoint and print `[RESUME]`.

---

## Evaluation

```bash
python evaluate.py \
    --img_dir data/ECSSD/images \
    --mask_dir data/ECSSD/masks \
    --ckpt checkpoints/best_improved.pth --improved
```

This prints test-set metrics and saves a 4-column visualization (input | ground truth | prediction | overlay).

---

## Demo

### Option 1 — Gradio web app

```bash
python app.py --ckpt checkpoints/best_improved.pth --share
```

Generates a public share link valid for 7 days.

### Option 2 — Interactive notebook

Open [demo_notebook.ipynb](demo_notebook.ipynb) in Jupyter or Google Colab.

---

## Method

**Architecture (improved model).** Encoder–decoder U-Net with 4 down-sampling stages, a 512-channel bottleneck, and 4 up-sampling stages. Each level has a Conv → BatchNorm → ReLU block. Encoder feature maps are concatenated into the matching decoder level (skip connections), which lets the decoder use both abstract semantic features and precise spatial detail.

- **Input:** 3 × 128 × 128 RGB
- **Output:** 1 × 128 × 128 sigmoid saliency map
- **Parameters:** 3.84 M

**Loss.** Binary cross-entropy + 0.5 × (1 − soft IoU). BCE alone tends to favor predicting "all background" early in training; adding the IoU term pulls predictions toward better region overlap.

**Optimization.** Adam (lr = 1e-3), batch size 16, up to 25 epochs with early stopping (patience = 5).

**Augmentation.** Horizontal flip, brightness jitter, and random crop — applied only to the training split. Geometric augmentations are applied identically to image and mask.

**Resumable training.** Every epoch saves `last.pth` (model weights, optimizer state, current epoch, history). On restart, the script detects the checkpoint and resumes seamlessly.

---

## Limitations and Next Steps

- Trained on ECSSD only (700 training images) — using DUTS-TR (~10,500) would substantially improve generalization.
- 128 × 128 resolution loses fine detail — moving to 224 × 224 with the same architecture is a one-line change.
- No pretrained backbone — initializing the encoder from VGG/ResNet would likely push IoU well past 0.75.
- Hard cases (camouflaged objects, foreground/background with similar colors) remain difficult — see the bee-on-flower example in the report.

---

## Author

**Dafina Keqmezi**
---
