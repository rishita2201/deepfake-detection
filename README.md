# Deepfake image detection: ResNet-18, ResNet-50, EfficientNet-B0 and Xception

Code, training histories, result tables and figures for the paper
**"Evaluating CNN Architectures for Deepfake Image Detection: Accuracy, Robustness, and Calibration"**
(R Rishita, Elakya R; Vellore Institute of Technology, Chennai).

Paper: under review  |  Model checkpoints: https://github.com/rishita123/deepfake-detection/releases/tag/v1.0

## What this repository contains
| Folder | Content |
|---|---|
| `notebooks/` | `01`-`04` train one backbone each (identical protocol); `05` evaluates and compares the four models (metrics, Grad-CAM, calibration, McNemar); `06` adds bootstrap CIs, validation-vs-test analysis, duplicate check, threshold tuning, full-test-set robustness, timing and Grad-CAM quantification |
| `history/` | Per-epoch loss and accuracy (JSON) of the four training runs (seed 42) |
| `results/tables/` | All result tables as CSV (`A_` dataset sizes ... `G_` Grad-CAM) plus `E4_flagged_test_images.csv`, the test images with a near-duplicate in train/val |
| `results/predictions/` | Saved fake-probabilities of every model on the validation and test sets (`*_seed42_{val,test}.npy`, sample order = notebook 06) |
| `results/robustness/` | Saved test-set probabilities under each degradation (`<name>_<level>.npy`, shape 4 models x 10,905) |
| `results/hashes/` | MD5 and 64-bit pHash of every dataset image, used by the duplicate check |
| `figures/` | Final paper figures (PNG, 300 dpi) |
| `scripts/` | `make_paper_figures.py` and `make_architecture_figure.py` regenerate the data-driven figures from `results/` and `history/` |
| `checkpoints/` | Only `SHA256SUMS.txt`; the `.pth` files are distributed separately (see below) |

## Main results (official test split, 10,905 images, threshold 0.5, one training run with seed 42)
| Model | Accuracy (%) | F1 | ROC-AUC | ECE | Validation acc. (%) |
|---|---|---|---|---|---|
| ResNet-18 | 90.57 | 0.9091 | 0.9668 | 0.0295 | 97.02 |
| ResNet-50 | 90.00 | 0.9069 | 0.9653 | 0.0511 | 97.99 |
| EfficientNet-B0 | 85.07 | 0.8672 | 0.9079 | 0.1142 | 98.45 |
| Xception | 89.90 | 0.9044 | 0.9592 | 0.0552 | 97.50 |

ResNet-18, ResNet-50 and Xception are statistically close; EfficientNet-B0 is clearly worse. Validation accuracy
overstates test accuracy by 6.4-13.4 points. These are single-seed results; confidence intervals in the paper reflect
test-set sampling only. See the paper for the full analysis.

## Dataset
Kaggle "Deepfake and Real Images" by Manjil Karki (a pre-processed subset of OpenForensics):
https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
The images are **not** included here; the dataset keeps its own license. Splits: train 70,001 real / 70,001 fake,
validation 19,787 / 19,641, test 5,413 / 5,492.

## How to reproduce
1. Open the notebooks in Google Colab (T4 GPU runtime). Each notebook downloads/unzips the dataset using your own `kaggle.json`
   (upload it to the Colab session; never commit it).
2. Run `01`-`04` (training), then `05` (evaluation), then `06` (extra analyses). The notebooks write to a Google Drive folder
   called `deepfake_results/`; adjust the paths in the first configuration cell if yours differ.
3. To redraw the figures from the saved results: `pip install -r requirements.txt`, then `python scripts/make_paper_figures.py`.

Protocol (all models): 224x224 input, ImageNet normalization, Adam with L2 weight decay 1e-4 (not AdamW), learning rate 1e-4
(constant), batch size 32, mixed precision, early stopping on validation loss (patience 2), best checkpoint restored, seed 42.
ResNet-50 uses torchvision `IMAGENET1K_V2` weights, ResNet-18 and EfficientNet-B0 use `IMAGENET1K_V1`, Xception comes from timm.
Environment: PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA Tesla T4.

## Checkpoints
The four best-epoch checkpoints (~230 MB in total) exceed GitHub's per-file limit and are published as release assets:
`resnet18_best.pth`, `resnet50_best.pth`, `efficientnet_b0_best.pth`, `xception_best.pth`.
Verify downloads with `sha256sum -c checkpoints/SHA256SUMS.txt`. Load them with the model builders in notebook `05`
(each checkpoint is a plain `state_dict`).

## Citation
See `CITATION.cff`. If you use this repository, please cite the paper (reference to be added after publication).

## License
Code: MIT (see `LICENSE`). The dataset and the pretrained ImageNet weights are subject to their own licenses.
