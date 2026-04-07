import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import io

class DogBreedPredictor:
    def __init__(self):
        # MLOps 파이프라인에서 빠른 추론을 위해 가벼운 MobileNetV3 모델을 사용합니다.
        self.weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=self.weights)
        self.model.eval()  # 추론 모드 설정
        self.transforms = self.weights.transforms()
        
        # ImageNet 클래스 1000개의 라벨 (151~268번이 개 품종입니다)
        self.categories = self.weights.meta["categories"]

    def predict(self, image_bytes: bytes) -> str:
        try:
            # 바이트 데이터를 PIL 이미지로 변환 및 RGB 처리
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # 모델 입력용 텐서로 변환: 전처리 및 배치 차원 추가
            tensor = self.transforms(image).unsqueeze(0)
            
            # 예측 수행
            with torch.no_grad():
                out = self.model(tensor)
            
            # 확률 계산
            probabilities = torch.nn.functional.softmax(out[0], dim=0)
            
            # 가장 높은 확률을 가진 클래스 추출 (ImageNet 기준 1000개 클래스 중)
            top_prob, top_catid = torch.topk(probabilities, 1)
            
            class_idx = top_catid.item()
            label = self.categories[class_idx]
            
            # ImageNet에서 강아지 클래스는 대략 151부터 268 범위를 가집니다.
            is_dog = 151 <= class_idx <= 268
            
            return {
                "label": label,
                "confidence": round(top_prob.item() * 100, 2),
                "is_dog_class": is_dog,
                "class_id": class_idx
            }
        except Exception as e:
            return {"error": str(e)}
