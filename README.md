# Sleep Apnea Detection

A machine learning pipeline for detecting sleep apnea events from polysomnography signals. Built as part of the SRIP 2026 application for the AI for Health project at IIT Gandhinagar under Prof. Nipun Batra.

---

## AI Tool Disclosure

This project was developed with assistance from Claude (Anthropic).

The signal preprocessing pipeline and visualizations — including the bandpass filtering, windowing, labeling strategy, and PDF generation — were written and understood by the author independently. These concepts were within the scope of what a second year undergraduate with a strong foundation in Python and data processing can grasp and implement.

For the deep learning components (1D CNN and Conv-LSTM), Claude was used as an aid. As a second year undergraduate, I have not yet covered neural networks and deep learning formally in my coursework. I understand the high level intuition behind the architecture — how convolutional layers detect local patterns, how LSTM adds sequential memory, and why Leave-One-Participant-Out evaluation is the right strategy for this problem. However, I used Claude to help translate that understanding into working PyTorch code.

I am fully committed to learning these concepts in depth before and during the internship period. This project has genuinely sparked my interest in health sensing and I look forward to building a much stronger foundation in deep learning over the coming months.

---

## Project Overview

Sleep apnea is a condition where breathing repeatedly stops during sleep. It affects millions of people worldwide and remains chronically underdiagnosed. This project builds an end to end pipeline that:

- Visualizes raw physiological signals from overnight sleep studies
- Preprocesses and segments signals into labeled 30 second windows
- Trains a 1D CNN and a Conv-LSTM model to classify breathing events
- Evaluates models using Leave-One-Participant-Out cross validation to ensure generalization to unseen individuals

---

## Dataset

The dataset contains overnight polysomnography recordings from 5 participants. Each participant folder contains 5 files:

File                                        Description                                               Sampling Rate 

`nasal_airflow.txt`             Nasal airflow signal                                                       32 Hz 
`thoracic_movement.txt`         Thoracic and abdominal movement                                            32 Hz 
`spo2.txt`                      Blood oxygen saturation                                                    4 Hz 
`flow_events.txt`               Annotated breathing events with start and end timestamps                     — 
`sleep_profile.txt`             Sleep stage annotations recorded every 30 seconds                            — 

---

## Project Structure

```
sleep-apnea-detection/
├── data/
│   ├── AP01/
│   │   ├── nasal_airflow.txt
│   │   ├── thoracic_movement.txt
│   │   ├── spo2.txt
│   │   ├── flow_events.txt
│   │   └── sleep_profile.txt
│   ├── AP02/
│   ├── AP03/
│   ├── AP04/
│   └── AP05/
├── Dataset/
│   ├── breathing_dataset.csv
│   ├── sleep_stage_dataset.csv
│   └── lopo_results.csv
├── models/
│   ├── cnn_model.py
│   └── conv_lstm_model.py
├── scripts/
│   ├── vis.ipynb
│   ├── create_dataset.ipynb
│   └── train_model.ipynb
├── Visualizations/
│   ├── AP01_visualization.pdf
│   ├── AP02_visualization.pdf
│   ├── AP03_visualization.pdf
│   ├── AP04_visualization.pdf
│   └── AP05_visualization.pdf
├── README.md
└── requirements.txt
```

---

## Setup

Clone the repository and install all dependencies:

```bash
git clone https://github.com/Adithya-Sirigiri/Sleep_apnea_detection.git
cd Sleep_apnea_detection
pip install -r requirements.txt
```

---

## Dependencies

 Library                                       Used For 

 `numpy`                         numerical operations and array manipulation
 `pandas`                        loading and processing signal data files 
 `scipy`                         butterworth bandpass filter for signal preprocessing
 `matplotlib`                    generating multi page PDF visualizations
 `scikit-learn`                  label encoding, normalization and evaluation metrics
 `torch`                         building and training the 1D CNN and Conv-LSTM models
 `PyWavelets`                    wavelet based signal processing utilities 

---

## How to Run

### Step 1 — Generate Visualizations

Open `scripts/vis.ipynb` in VS Code and run all cells.

Change `participant_path` in Cell 2 to switch between patients:

```python
participant_path = "../data/AP01"  # change AP01 to AP02, AP03 etc
```

Output PDFs are saved to the `Visualizations/` folder.

### Step 2 — Create Dataset

Open `scripts/create_dataset.ipynb` and run all cells.

This generates two files in the `Dataset/` folder:
- `breathing_dataset.csv` — windows labeled by breathing event type
- `sleep_stage_dataset.csv` — windows labeled by sleep stage

### Step 3 — Train Models

Open `scripts/train_model.ipynb` and run all cells.

This trains the 1D CNN using Leave-One-Participant-Out cross validation and prints accuracy, precision, recall and confusion matrix for each fold and overall.

---

## Signal Preprocessing

All signals are preprocessed before windowing:

- Butterworth bandpass filter applied to isolate the breathing frequency range of 0.17 Hz to 0.4 Hz
- Human breathing occurs at 10 to 24 breaths per minute which corresponds to this frequency range
- Frequencies outside this range are noise from body movement and sensor interference

Windowing parameters:

| Signal | Sampling Rate | Samples per Window | Step Size |
|--------|--------------|-------------------|-----------|
| Nasal Airflow | 32 Hz | 960 | 480 (50% overlap) |
| Thoracic Movement | 32 Hz | 960 | 480 (50% overlap) |
| SpO2 | 4 Hz | 120 | 60 (50% overlap) |

---

## Labeling Strategy

### breathing_dataset.csv
Each 30 second window is labeled using the flow events file:
- If a breathing event overlaps more than 50% of the window duration → assign that event label (Hypopnea or Obstructive Apnea)
- If no event overlaps more than 50% → label as Normal

### sleep_stage_dataset.csv
Each 30 second window is labeled using the sleep profile file:
- Sleep stage (Wake, N1, N2, N3, REM) assigned based on the closest timestamp entry in the sleep profile

---

## Models

### 1D CNN (`models/cnn_model.py`)

Three convolutional blocks with increasing filter sizes:

```
Input (batch, 3, 960)
    → Conv1d(3→32, kernel=7) + BatchNorm + ReLU + MaxPool
    → Conv1d(32→64, kernel=5) + BatchNorm + ReLU + MaxPool
    → Conv1d(64→128, kernel=3) + BatchNorm + ReLU + MaxPool
    → AdaptiveAvgPool1d(1)
    → Linear(128→64) + ReLU + Dropout(0.5)
    → Linear(64→3)
Output: class scores for Normal, Hypopnea, Obstructive Apnea
```

### Conv-LSTM (`models/conv_lstm_model.py`)

Same convolutional frontend as the CNN followed by an LSTM layer:

```
Input (batch, 3, 960)
    → Conv blocks (same as CNN above)
    → Reshape for LSTM (batch, time_steps, 128)
    → LSTM(input=128, hidden=64, layers=2)
    → Final hidden state
    → Linear(64→32) + ReLU + Dropout(0.5)
    → Linear(32→3)
Output: class scores for Normal, Hypopnea, Obstructive Apnea
```

The Conv-LSTM captures both local breathing patterns through the convolutional layers and sequential memory across the window through the LSTM.

---

## Evaluation Strategy

Leave-One-Participant-Out (LOPO) cross validation:

```
Fold 1: Train on AP02 AP03 AP04 AP05 → Test on AP01
Fold 2: Train on AP01 AP03 AP04 AP05 → Test on AP02
Fold 3: Train on AP01 AP02 AP04 AP05 → Test on AP03
Fold 4: Train on AP01 AP02 AP03 AP05 → Test on AP04
Fold 5: Train on AP01 AP02 AP03 AP04 → Test on AP05
```

This ensures the model is always tested on a participant it has never seen during training, simulating real world deployment on new patients.

Metrics reported per fold and overall:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- Confusion Matrix
- Classification Report

---

## Results

Per fold results are saved to `Dataset/lopo_results.csv` 
after running `train_model.ipynb`.
For detailed analysis and discussion see `report.pdf`.

---

## Author

Sirigiri Venkateswara Adithya
B.Tech CSE (Data Science and Analytics), 2nd Year
IIIT Sonepat | CPI: 9.43