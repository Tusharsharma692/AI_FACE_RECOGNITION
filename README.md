# 🔍 Face Verification System (FastAPI + InsightFace)

A web-based Face Verification system that compares two uploaded images
and determines whether they belong to the same person using deep
learning.

------------------------------------------------------------------------

## 🚀 Features

-   Upload two images
-   Automatic face detection
-   Deep face embedding extraction
-   Cosine similarity comparison
-   Displays similarity score
-   Shows uploaded images
-   Clean and responsive UI layout

------------------------------------------------------------------------

## 🧠 How It Works

1.  User uploads two images.
2.  Backend saves images securely.
3.  InsightFace extracts 512-dimensional face embeddings.
4.  Embeddings are normalized using NumPy.
5.  Cosine similarity is calculated using Scikit-learn.
6.  Result is dynamically rendered using Jinja2 templates.

If similarity \> threshold → ✅ Same Person\
Else → ❌ Different Person

------------------------------------------------------------------------

## 📂 Project Structure

face-verification/ │ ├── main.py ├── requirements.txt ├── README.md │
├── static/ │ ├── style.css │ ├── logo.png │ └── uploads/ │ └──
templates/ └── index.html

------------------------------------------------------------------------

## ⚙️ Installation

### 1️⃣ Clone the repository

git clone `<your-repo-link>`{=html}\
cd face-verification

### 2️⃣ Create Virtual Environment (Recommended)

python -m venv venv

Activate:

Windows: venv`\Scripts`{=tex}`\activate`{=tex}

Linux/Mac: source venv/bin/activate

### 3️⃣ Install Dependencies

pip install -r requirements.txt

------------------------------------------------------------------------

## ▶️ Run the Application

uvicorn main:app --reload

Open in browser: http://127.0.0.1:8000

------------------------------------------------------------------------

## 📊 Model Used

-   InsightFace model: buffalo_l
-   Embedding size: 512
-   Similarity metric: Cosine Similarity

------------------------------------------------------------------------

## 🛠 Technologies Used

-   FastAPI
-   Uvicorn
-   InsightFace
-   OpenCV
-   NumPy
-   Scikit-learn
-   Jinja2
-   ONNX Runtime

------------------------------------------------------------------------

## 🎯 Future Improvements

-   Multi-face comparison support
-   Bounding box visualization
-   REST API version
-   Cloud deployment (AWS / Render / Railway)
-   Authentication system

------------------------------------------------------------------------

## 👨‍💻 Author

Tushar Sharma\
AI & Deep Learning Enthusiast 🚀
