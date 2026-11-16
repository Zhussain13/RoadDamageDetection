# RoadCare+

**A Lightweight YOLOv7 Framework Enhanced with GhostNet, BasicRFB and CARAFE for Road Surface Damage Detection in India**

[![Scopus](https://img.shields.io/badge/Scopus-indexed-orange)](https://www.scopus.com/record/display.uri?eid=2-s2.0-105047263229&origin=resultslist)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-YOLOv7-red)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)

RoadCare+ detects potholes and cracks in road videos in real time. It improves YOLOv7 with three
modules (the variant is called **GRC-YOLOv7**) so that it stays accurate while being small and
fast enough for low-cost hardware, for use by municipalities and smart-city road maintenance.

---

## Publication

The work is **published and indexed in Scopus**.

| | |
|---|---|
| **Title** | RoadCare+: A Lightweight YOLOv7 Framework Enhanced with GhostNet, BasicRFB and CARAFE for Road Surface Damage Detection in India |
| **Venue** | ACT 2026 |
| **Scopus EID** | `2-s2.0-105047263229` |
| **Scopus record** | https://www.scopus.com/record/display.uri?eid=2-s2.0-105047263229&origin=resultslist |

If you use this work, please cite it (see [`CITATION.cff`](CITATION.cff); GitHub's **Cite this repository**
button gives APA and BibTeX). Full author list, volume and pages are on the Scopus record.

---

## Highlights

| Module | What it adds |
|---|---|
| **GhostNet** | Cheap "ghost" feature maps → fewer parameters and faster inference |
| **BasicRFB** (Receptive Field Block) | Wider context → better at thin cracks and small potholes |
| **CARAFE** (content-aware upsampling) | Sharper multi-scale features in the neck |

- Trained on the **RDD2020 India subset** (Sekilab) for Indian road conditions.
- Side-by-side **Streamlit app**: baseline YOLOv7 vs the customised model on the same video.
- Pretrained weights and a test video included, so the demo runs without training.

## Results (RDD2020 India test split)

| Metric | Value |
|---|---|
| mAP@0.5 | **70.6%** |
| Precision | 0.71 |
| Recall | 0.69 |
| F1-score | 0.70 |
| Speed | ~35 FPS (CPU), ~62 FPS (GPU) |
| Model size | ~12.5 MB |

## Damage classes (RDD2020)

| Code | Damage type |
|---|---|
| D00 | Longitudinal crack |
| D10 | Transverse crack |
| D20 | Alligator crack |
| D40 | Pothole |

> The demo weights in `Trained_Models/` are labelled by `classes.names`, which currently lists one
> class (`pothole`).

---

## Repository contents

```
RoadDamageDetection/
├── Main.py                     Streamlit app: baseline YOLOv7 vs customised YOLOv7, side by side
├── requirements.txt            Python packages
├── CITATION.cff                How to cite the published paper
├── Trained_Models/
│   ├── pothole_detection_basicyolov7.pt      baseline YOLOv7 weights
│   ├── Pothole_detection_yolov7customised.pt customised (RoadCare+) weights
│   └── classes.names
├── Test_data/
│   └── test_video_1.mp4        sample road video
└── Docs/
    ├── Final Copy.pdf          project report (M.Tech mini project)
    ├── Mini Project.pptx       presentation
    └── *.pdf                   reference papers (RDD2022 dataset, related methods)
```

---

## How to run the demo

**1. Get the code and install packages** (Python 3.8+):

```bash
git clone https://github.com/Zhussain13/RoadDamageDetection.git
cd RoadDamageDetection
pip install -r requirements.txt
```

**2. Get the YOLOv7 code** used for the baseline model (it provides `models.experimental` and
`utils.general`). Clone it next to `Main.py` in a folder named `yolobasic_v7`:

```bash
git clone https://github.com/WongKinYiu/yolov7.git yolobasic_v7
```

**3. Set the paths at the top of `Main.py`.** They currently point to the author's Windows folders
(`D:\TA\M.Tech\...`). Change these four lines to your copy, for example:

```python
YOLObasic_v7_PATH = "yolobasic_v7"
YOLOv7_Customised_MODEL_PATH = "Trained_Models/Pothole_detection_yolov7customised.pt"
YOLObasic_v7_MODEL_PATH = "Trained_Models/pothole_detection_basicyolov7.pt"
NAMES_PATH = "Trained_Models/classes.names"
```

**4. Start the app** (it is a Streamlit app, so use `streamlit run`, not `python`):

```bash
streamlit run Main.py
```

Open the link it prints (usually http://localhost:8501), upload `Test_data/test_video_1.mp4` or any
road video, and watch both models' detections frame by frame.

---

## Method in brief

1. **Data:** RDD2020 India images, YOLO-format labels, resizing and augmentation.
2. **Model:** YOLOv7 backbone and neck modified with GhostNet, BasicRFB and CARAFE (GRC-YOLOv7).
3. **Training:** PyTorch, SGD with cosine learning-rate schedule, early stopping, mixed precision.
4. **Inference:** letterbox resize → model → confidence threshold 0.25 → NMS (IoU 0.45) → boxes on video.

Full details, ablations and comparisons are in [`Docs/Final Copy.pdf`](Docs/Final%20Copy.pdf).

## Applications

Municipal road-maintenance planning · smart-city monitoring · dash-cam and vehicle-mounted
inspection · accident prevention · input for autonomous-driving perception.

---

## Author

**Shaik Mohammed Zaheed Hussain**, Assistant Professor, Dept. of CSE, CMR Institute of Technology,
Bengaluru. [LinkedIn](https://www.linkedin.com/in/zhussain13)

Developed as an M.Tech (Computer Science) mini project at Kishkinda University, Ballari (2024–25),
under the guidance of **Dr. Rajashree V. Biradar**, Chairperson, Dept. of CSE.

## Acknowledgements

RDD2020 dataset by Sekilab; YOLOv7 by Wang, Bochkovskiy and Liao; GhostNet, RFB and CARAFE by their
respective authors.

## License

Academic and research use. Please contact the author for any other use.
