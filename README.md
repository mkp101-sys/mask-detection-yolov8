<div align="center">

# 😷 Mask vs No-Mask Detection — YOLOv8

### Real-time face mask detection, fine-tuned, benchmarked, and deployment-ready.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-00FFFF?style=flat&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![Ultralytics](https://img.shields.io/badge/Framework-Ultralytics-111F68?style=flat)](https://github.com/ultralytics/ultralytics)
[![ONNX](https://img.shields.io/badge/Export-ONNX-005CED?style=flat&logo=onnx&logoColor=white)](https://onnx.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

<img src="/before_after_comparison.png" width="90%" />

</div>

<br>

## 📖 Table of Contents

- [Overview](#-overview)
- [Results at a Glance](#-results-at-a-glance)
- [Visual Results](#-visual-results)
- [Confidence Threshold Tradeoff](#-confidence-threshold-tradeoff)
- [Inference Speed](#-inference-speed)
- [Group-Photo Compliance Count](#-group-photo-compliance-count)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Limitations](#-limitations)
- [License](#-license)

<br>

## 🎯 Overview

This project fine-tunes a pretrained **YOLOv8** object detector to classify faces as **Mask** or **NO-Mask** in real time, then goes a step further — counting mask compliance across an entire group of people in a single pass.

> **Note:** This is a fine-tuned model, not trained from scratch. Base YOLOv8 weights are pretrained on COCO; only the detection head/backbone are fine-tuned here on the mask dataset. Full transparency on scope in [Limitations](#-limitations).

**What makes this more than a tutorial copy-paste:**
- ✅ Proper train / validation / test evaluation — not just visual inspection
- ✅ Two model sizes compared head-to-head (YOLOv8n vs YOLOv8s)
- ✅ Confidence threshold tuned and justified with data, not guessed
- ✅ Real measured inference speed on both GPU and CPU
- ✅ Extended beyond single-face detection to multi-person compliance counting
- ✅ Exported to ONNX for lightweight deployment

<br>

## 📊 Results at a Glance

Evaluated on a held-out **test set** — 353 images, 547 face instances:

| Model       | Precision | Recall | mAP50     | mAP50-95 | Params    |
|:-----------:|:---------:|:------:|:---------:|:--------:|:---------:|
| **YOLOv8n** | **93.7%** | 88.2%  | **92.7%** | 64.6%    | **3.0M**  |
| YOLOv8s     | 93.5%     | 87.3%  | 92.0%     | 64.4%    | 11.1M     |

🏆 **YOLOv8n wins** — essentially matching YOLOv8s on accuracy at a third of the parameters and faster inference. Bigger isn't always better.

<br>

## 🖼️ Visual Results

<table>
<tr>
<td align="center" width="50%"><b>Confusion Matrix</b><br><img src="/confusion_matrix.png" width="100%"/></td>
<td align="center" width="50%"><b>Precision–Recall Curve</b><br><img src="/pr_curve.png" width="100%"/></td>
</tr>
</table>

<br>

## ⚖️ Confidence Threshold Tradeoff

Sweeping the confidence threshold reveals the classic precision/recall tension:

| Threshold | Precision | Recall | mAP50 |
|:---------:|:---------:|:------:|:-----:|
| 0.10      | 93.7%     | 88.2%  | 90.9% |
| **0.25**  | **93.7%** | **88.2%** | **89.4%** |
| 0.50      | 94.4%     | 87.0%  | 86.2% |
| 0.70      | 98.0%     | 77.3%  | 76.5% |

📌 **0.25 is used as the default** — it catches nearly everyone (88% recall) while keeping precision high. Push the threshold to 0.70 and precision hits 98%, but you start missing ~1 in 4 people — a bad tradeoff for a compliance-monitoring use case.

<br>

## ⚡ Inference Speed

| Device            | FPS       | ms / frame |
|:------------------:|:---------:|:----------:|
| 🖥️ GPU (Tesla T4)  | **113.7** | 8.8        |
| 💻 CPU              | 8.8       | 113.8      |

A **~13x speed gap** between GPU and CPU — a solid reminder that "runs fine in the notebook" and "runs fine in production" are two different conversations.

<br>

## 👥 Group-Photo Compliance Count

Tested on a 20-person photo grid — detected **all 20/20 people** correctly:

<div align="center">

**Mask: 12** &nbsp;|&nbsp; **NO-Mask: 8**

</div>

<br>

## 🗂️ Dataset

[`mask-wearing-iskms`](https://universe.roboflow.com/roboflow-universe-projects/mask-wearing-iskms) via Roboflow Universe
— 2 classes (`Mask`, `NO-Mask`), ~8,400 images across train / val / test splits.

Full credit to the original dataset creators — this project does not claim ownership of the data.

<br>

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Ultralytics](https://img.shields.io/badge/-Ultralytics%20YOLOv8-111F68?style=flat-square)
![Roboflow](https://img.shields.io/badge/-Roboflow-6706CE?style=flat-square)
![ONNX](https://img.shields.io/badge/-ONNX-005CED?style=flat-square)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square)
![Gradio](https://img.shields.io/badge/-Gradio-F97316?style=flat-square)

</div>

<br>

## 🚀 Getting Started

1. Open `mask_detection_yolov8.ipynb` in **Google Colab** (recommended — uses Colab Secrets + free GPU).
2. Set the runtime to GPU: `Runtime → Change runtime type → GPU`.
3. Add your Roboflow API key as a Colab secret named `ROBOFLOW_API_KEY` (🔑 icon in the left sidebar).
   > ⚠️ Never hardcode API keys directly in the notebook.
4. Run all cells top to bottom. Each section is clearly labeled:
   - Secure setup & dataset download
   - Training YOLOv8n
   - Confusion matrix & PR curve visualization
   - YOLOv8n vs YOLOv8s comparison
   - Confidence-threshold sweep
   - CPU vs GPU speed benchmark
   - Single-image test + ONNX export
   - Group-photo detection & counting
   - Optional Gradio demo

### Run locally instead

```bash
pip install -r requirements.txt
jupyter notebook mask_detection_yolov8.ipynb
```

> Local runs need `ROBOFLOW_API_KEY` set as an environment variable, and the Colab-specific cells (secrets / file upload) adapted for your environment.

<br>

## ⚠️ Limitations

- Dataset is COVID-era and relatively small — may not generalize well to side profiles, low light, or newer mask styles (patterned masks, face shields).
- Class balance between Mask / NO-Mask was checked but not explicitly corrected for.
- Tested on static images only, not video streams.
- Not intended for safety-critical or compliance-enforcement deployment without further validation on more diverse data.

<br>

## 👤 Author
 
**Maharshi K Patel**

DAIICT (Dhirubhai Ambani Institute of Information and Communication Technology) · IIRS (Indian Institute of Remote Sensing)

## 📄 License

Licensed under the [MIT License](LICENSE). The dataset itself is subject to its own license on Roboflow Universe.

<br>

<div align="center">

If this was useful, consider ⭐ starring the repo!

</div>
