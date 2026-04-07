from fastapi import FastAPI, File, UploadFile, HTTPException
from app.model import DogBreedPredictor
import uvicorn
import time

app = FastAPI(
    title="Dog Breed Predictor API",
    description="MLOps Pipeline - FastAPI Server for Dog Breed Prediction using MobileNetV3",
    version="1.0.0"
)

# 모델 생성 (서버 시작 시 메모리에 로드 - Singleton)
print("Loading model...")
predictor = DogBreedPredictor()
print("Model loaded successfully!")

@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "Welcome to the Dog Breed Prediction API. Use /predict to classify dogs."
    }

@app.post("/predict")
async def predict_dog_breed(file: UploadFile = File(...)):
    # 파일 확장자 확인 (간단한 이미지 검증)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (e.g. jpeg, png).")
    
    start_time = time.time()
    
    # 이미지 파일 읽기
    content = await file.read()
    
    # 모델을 통해 추론
    prediction = predictor.predict(content)
    
    # 에러가 발생한 경우
    if "error" in prediction:
        raise HTTPException(status_code=500, detail=prediction["error"])
        
    inference_time = round((time.time() - start_time) * 1000, 2)
    
    return {
        "filename": file.filename,
        "prediction": prediction,
        "processing_time_ms": inference_time
    }

if __name__ == "__main__":
    # 개발 서버 실행
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
