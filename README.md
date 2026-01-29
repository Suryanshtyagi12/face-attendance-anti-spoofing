Face Attendance System with Anti-Spoofing

A face attendance system that performs live face recognition with challenge–response anti-spoofing, incremental user registration, threshold-based matching, and audit-ready logging, ensuring secure and reliable attendance marking.

📌 Project Overview

This project implements an end-to-end face attendance pipeline that:

Registers users without retraining

Performs live face recognition

Prevents spoofing using liveness challenges

Marks attendance with punch-in / punch-out logic

Logs all system events for auditability

The system is designed to be:

Lightweight

Secure

Reproducible

Interview-ready

🔁 High-Level Workflow
User Registration
      ↓
Embedding Storage
      ↓
Live Camera Input
      ↓
Face Detection & Embedding Extraction
      ↓
Face Matching (Threshold-based)
      ↓
Challenge–Response Anti-Spoofing
      ↓
Attendance Decision
      ↓
CSV Storage + System Logs

🗂️ Folder Structure
face_attendance/
│
├── data/
│   ├── embeddings/           # Stored face embeddings (.npy)
│   └── attendance.csv        # Attendance records (runtime)
│
├── logs/
│   └── system.log            # Runtime logs (not committed)
│
├── research/
│   └── notebooks/
│       └── threshold_analysis.ipynb
│
├── register_face.py          # User registration
├── recognize_face.py         # Recognition + spoof + attendance
├── spoof_check.py            # Anti-spoofing logic
├── attendance.py             # Punch-in / punch-out logic
├── utils.py                  # Embedding & distance utilities
├── logger.py                 # Central logging config
│
├── requirements.txt
└── README.md

⚙️ Core Components
🟢 1. User Registration

Captures 15 face samples via webcam

Generates embeddings using pretrained models

Stores embeddings as .npy

No retraining or downtime

File: register_face.py

🟢 2. Face Recognition

Live webcam capture

Face detection + embedding extraction

Distance-based matching (Euclidean)

Configurable threshold

File: recognize_face.py

🟢 3. Anti-Spoofing (Liveness Check)

Triggered only after recognition.

Supported challenges:

Eye blink (EAR-based)

Head turn (left / right)

Features:

Temporal validation (multi-frame)

Prevents static photo/video attacks

CPU-only (no extra hardware)

File: spoof_check.py

🟢 4. Attendance System

One punch-in + one punch-out per user per day

CSV-based storage

Duplicate attempts are ignored and logged

File: attendance.py

🟢 5. Logging System

Centralized logging using Python logging

Logs:

Registration events

Recognition attempts

Spoof failures

Attendance actions

Errors and warnings

No print() statements in final system

File: logger.py

🔐 Security & Design Decisions
Decision	Justification
Pretrained embeddings	No training cost, faster onboarding
Incremental registration	No system downtime
Challenge–response spoofing	Prevents basic spoof attacks
CSV + logs	Lightweight & auditable
CPU-only	Works on low-resource systems
📊 Threshold Justification

Distance threshold is:

Empirically analyzed

Compared using Euclidean vs Cosine

Documented in research notebook

📁 research/notebooks/threshold_analysis.ipynb

🧪 How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Register a user
python register_face.py

3️⃣ Run full system
python recognize_face.py



🚫 What Is Intentionally Not Included

Cloud deployment

Deep liveness models

GPU dependencies

Centralized databases

This keeps the system simple, explainable, and reproducible.

📌 Future Improvements

Streamlit web interface

Database backend

Multi-camera support

Stronger liveness models

Role-based admin panel

👤 Author

Suryansh
AI / ML Engineering