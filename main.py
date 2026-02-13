from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import cv2
import numpy as np
import uuid
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/compare", response_class=HTMLResponse)
async def compare_faces(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    # Unique filenames (avoid cache overwrite)
    filename1 = str(uuid.uuid4()) + "_" + file1.filename
    filename2 = str(uuid.uuid4()) + "_" + file2.filename

    path1 = os.path.join(UPLOAD_FOLDER, filename1)
    path2 = os.path.join(UPLOAD_FOLDER, filename2)

    with open(path1, "wb") as buffer:
        shutil.copyfileobj(file1.file, buffer)

    with open(path2, "wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    faces1 = face_app.get(img1)
    faces2 = face_app.get(img2)

    if len(faces1) == 0 or len(faces2) == 0:
        result = "⚠ No face detected in one of the images."
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": result}
        )

    emb1 = faces1[0].embedding.reshape(1, -1)
    emb2 = faces2[0].embedding.reshape(1, -1)

    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)

    similarity = cosine_similarity(emb1, emb2)[0][0]

    threshold = 0.4

    if similarity > threshold:
        result = f"✅ Same Person (Similarity: {similarity:.2f})"
    else:
        result = f"❌ Different Person (Similarity: {similarity:.2f})"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "image1_url": f"/static/uploads/{filename1}",
            "image2_url": f"/static/uploads/{filename2}"
        }
    )
