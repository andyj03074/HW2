# PyTorch의 용량을 줄이기 위해 slim 버전의 파이썬 이미지를 베이스로 사용합니다.
FROM python:3.11-slim

# 작업 디렉터리 설정
WORKDIR /app

# 시스템 의존성 설치 (필요없는 캐시 삭제로 이미지 경량화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 (Docker 빌드 캐시 최적화를 위해 먼저 복사)
COPY requirements.txt .

# 1초라도 빠른 빌드와 이미지 용량 절감을 위해 pytorch CPU 버전 설치
# (기본 torch를 설치하면 CUDA 용량 약 2GB가 추가됨)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 나머지 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY ./app ./app

# FastAPI 포트 개방
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
