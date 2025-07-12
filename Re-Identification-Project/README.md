# 🚗 Vehicle Re-Identification

## ✅ Requirements

- **Python:** 3.9  
- **Virtual Environment:** Recommended  
- **System Dependencies:**

```bash
sudo apt update
sudo apt install ffmpeg
```

---

## 🛠️ Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Clone the Deep-Person-ReID Repository

```bash
git clone https://github.com/KaiyangZhou/deep-person-reid.git
```

### 3. Install Deep-Person-ReID

```bash
cd deep-person-reid
pip install -r requirements.txt
python3 setup.py develop
```

---

## 🚀 How to Run

### 🔁 If You Want to Train the Models from Scratch

#### 4. Prepare the Dataset for Roboflow

```bash
python3 roboflow_dataset_preparation.py
```

#### 5. Prepare the Dataset for YOLO

```bash
python3 yolo_dataset_preparation.py
```

#### 6. Train the YOLO Model

```bash
python3 train_yolo.py
```

#### 7. Prepare the Dataset for ReID

```bash
python3 reid_dataset_preparation.py
```

#### 8. Train the ReID Model

```bash
python3 train_reid.py
```

#### 9. Compute the Best Matching Threshold

```bash
python3 compute_best_threshold.py
```

> 📌 **Important:**  
> Open `query.py` and update the threshold variable with the best value obtained in **step 9**.

---

### ✅ If You Already Have Trained Models

#### 11. Run the Inference Pipeline

```bash
python3 main.py --input_video_path /path/to/input_video --query_video_path /path/to/query_video
```