# Face Recognition System 🧠📸

This project implements a face recognition system using the [`face_recognition`](https://github.com/ageitgey/face_recognition) library and OpenCV. It can detect and recognize multiple faces in real-time using a webcam or image files.

---

## 🚀 How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Landmark Model

You need the `shape_predictor_68_face_landmarks.dat` file from Dlib.  
You have two options:

#### Option A – Automatic download:

```bash
python download_model.py
```

#### Option B – Manual download:

1. Download from here:  
   [Dlib model download](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
2. Extract and place the `.dat` file in the root project directory.

---

### 3. Encode Known Faces

```bash
python train_model.py
```

This will generate `encodings.pickle` storing face encodings for all people in `dataset/`.

---

### 4. Run Face Recognition

```bash
python recognize.py
```

This script will access your webcam and recognize known faces in real time.

---
## 🖼️ Sample Output

The system predicts the person and displays a confidence score in real-time:

![Prediction Example](linkedin_recognition.png)

---
## 📦 Dependencies

- `face_recognition`
- `dlib`
- `opencv-python`
- `numpy`
- `imutils` *(if used for resizing or rotation)*

See `requirements.txt` for full list.

---


