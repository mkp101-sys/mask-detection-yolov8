# 😷 Mask vs No-Mask Detection — YOLOv8

A real-time face mask detector built by fine-tuning **YOLOv8** on a public dataset, with full evaluation, a model-size comparison, a confidence-threshold tradeoff study, an inference speed benchmark, and group-photo compliance counting.

![Detection demo](results/before_after_comparison.png)

## 📌 Overview

This project fine-tunes a pretrained YOLOv8 object detector to classify faces as **Mask** or **NO-Mask**, then extends single-image detection to **count compliance across a group of people** — useful for lightweight monitoring use cases (e.g. entry checks).

**This is a fine-tuned model, not trained from scratch.** The base YOLOv8 weights are pretrained on COCO; only the detection head and backbone are fine-tuned here on the mask dataset. Full transparency on scope and limitations is in the [Limitations](#-limitations) section below.

## 📊 Results

Evaluated on a held-out **test set** (353 images, 547 face instances):

| Model   | Precision | Recall | mAP50  | mAP50-95 | Params |
|---------|-----------|--------|--------|----------|--------|
| YOLOv8n | 93.7%     | 88.2%  | 92.7%  | 64.6%    | 3.0M   |
| YOLOv8s | 93.5%     | 87.3%  | 92.0%  | 64.4%    | 11.1M  |

**YOLOv8n was chosen for deployment** — essentially the same accuracy as YOLOv8s at a third of the parameters and faster inference.

<p align="center">
  <img src="results/confusion_matrix.png" width="45%" />
  <img src="results/pr_curve.png" width="45%" />
</p>

### Confidence threshold tradeoff

| Threshold | Precision | Recall | mAP50 |
|-----------|-----------|--------|-------|
| 0.10      | 93.7%     | 88.2%  | 90.9% |
| 0.25      | 93.7%     | 88.2%  | 89.4% |
| 0.50      | 94.4%     | 87.0%  | 86.2% |
| 0.70      | 98.0%     | 77.3%  | 76.5% |

Lower thresholds catch more faces (higher recall) at the cost of more false positives; higher thresholds are stricter but miss more people. 0.25 is used as the default in this repo.

### Inference speed

| Device            | FPS   | ms / frame |
|-------------------|-------|------------|
| GPU (Tesla T4)    | 113.7 | 8.8        |
| CPU               | 8.8   | 113.8      |

### Group-photo compliance count

Tested on a 20-person photo grid — correctly detected **20/20** people (**12 Mask / 8 NO-Mask**).

## 🗂️ Dataset

[`mask-wearing-iskms`](https://universe.roboflow.com/roboflow-universe-projects/mask-wearing-iskms) by Roboflow Universe — 2 classes (`Mask`, `NO-Mask`), ~8,400 images total across train/val/test splits. Credit to the original dataset creators; this project does not claim ownership of the data.

## 🛠️ Tech stack

- Python
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Roboflow (dataset hosting/download)
- ONNX (deployment export)
- Pandas / Matplotlib (analysis & plotting)
- Gradio (optional demo UI)

## 🚀 Getting started

1. Clone this repo and open `mask_detection_yolov8.ipynb` in **Google Colab** (recommended — uses Colab Secrets and GPU runtime).
2. Set Colab's runtime to GPU: `Runtime → Change runtime type → GPU`.
3. Add your Roboflow API key as a Colab secret named `ROBOFLOW_API_KEY` (key icon in the left sidebar). **Never hardcode API keys in the notebook.**
4. Run the cells top to bottom. Each section is labeled and covers:
   - Secure setup & dataset download
   - Training YOLOv8n
   - Confusion matrix & PR curve visualization
   - YOLOv8n vs YOLOv8s comparison
   - Confidence-threshold sweep
   - CPU vs GPU speed benchmark
   - Single-image test + ONNX export
   - Group-photo detection & counting (before/after visualization)
   - Optional Gradio demo

To run locally instead of Colab:

```bash
pip install -r requirements.txt
jupyter notebook mask_detection_yolov8.ipynb
```

(Local runs need `ROBOFLOW_API_KEY` set as an environment variable, and the Colab-specific cells — secrets/file upload — adapted for your environment.)

## ⚠️ Limitations

- Dataset is COVID-era and relatively small — may not generalize well to side profiles, low light, or newer mask styles (e.g. patterned masks, face shields).
- Class balance between Mask / NO-Mask was checked but not explicitly corrected for.
- Tested on static images only, not video streams.
- Not intended for safety-critical or compliance-enforcement deployment without further validation on more diverse data.

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details. The dataset itself is subject to its own license on Roboflow Universe.
