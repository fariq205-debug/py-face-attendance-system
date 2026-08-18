# ⚡ VisionAuth — Enterprise AI Attendance Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3.0-003B57.svg)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**VisionAuth** is a smart, real-time AI facial recognition attendance system designed for seamless access control and duration tracking. Built with **OpenCV (LBPH Recognizer)**, **Streamlit**, and **SQLite**, it provides dual-role access control, live facial sampling, automatic check-in/check-out time calculations, and real-time analytical dashboards.

---

## ✨ Key Features

- **🔐 Dual-Role Access Control:** Separate login interfaces for `Admin` (full control & candidate management) and `Kiosk Operator` (live scanner interface).
- **📸 Automatic Facial Sampling:** Enrolls new candidates by capturing and preprocessing 20 image samples automatically.
- **⚡ Real-Time Face Recognition:** Employs Haar Cascade detectors and OpenCV's **LBPH (Local Binary Patterns Histograms)** recognizer for fast, accurate matching.
- **⏱️ Automated Check-In & Check-Out Tracking:**
  - *First Detection:* Records candidate **Check-In** time.
  - *Subsequent Detections:* Updates candidate **Check-Out** time continuously.
  - *Automatic Duration Math:* Dynamically calculates total stay duration (e.g., `2h 15m`, `45m 12s`).
- **📊 Executive Analytics Dashboard:** Live metrics displaying total enrolled candidates, present count, and absent count for the day.
- **🗃️ Built-In SQLite Database:** Zero-configuration local database with automatic schema migrations.

---

## 🛠️ Tech Stack

- **Frontend / Dashboard UI:** Streamlit
- **Computer Vision:** OpenCV (`cv2`), OpenCV-Contrib (`cv2.face`)
- **Machine Learning Algorithm:** LBPH Face Recognizer
- **Database:** SQLite3
- **Data Handling:** Pandas, NumPy
- **Language:** Python 3.9+

---

## 📂 Project Structure
vision-auth-py/
├── app.py                          # Main Streamlit Application Entrypoint
├── attendance.db                   # SQLite Database (Auto-generated)
├── haarcascade_frontalface_default.xml # Face Detection Cascade Classifier
├── student_images/                 # Captured Face Dataset Directory
└── README.md                       # Project Documentation
git clone [https://github.com/YOUR_USERNAME/vision-auth-py.git](https://github.com/YOUR_USERNAME/vision-auth-py.git)
cd vision-auth-py
