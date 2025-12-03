from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import uvicorn
import os

# FastAPI 앱 생성
app = FastAPI(
    title="감정 분류 API",
    description="한국어 텍스트의 감정을 6가지로 분류하는 API",
    version="1.0.0",
    docs_url="/textEmotion/docs",   # Swagger UI 경로 변경
    redoc_url="/textEmotion/redoc", # ReDoc 경로 변경 (원하면)
    openapi_url="/textEmotion/openapi.json"  # OpenAPI 스펙 경로
)

# 전역 변수로 모델과 토크나이저 저장
model = None
tokenizer = None
device = None

# 감정 라벨 매핑
EMOTION_LABELS = {
    0: "라벨 1",
    1: "라벨 2", 
    2: "라벨 3",
    3: "라벨 4",
    4: "라벨 5",
    5: "라벨 6"
}


# 요청 모델
class TextRequest(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "오늘 정말 기분이 좋아요!"
            }
        }


class BatchTextRequest(BaseModel):
    texts: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "오늘 정말 기분이 좋아요!",
                    "너무 슬프고 우울해요"
                ]
            }
        }


# 응답 모델
class EmotionResponse(BaseModel):
    text: str
    predicted_label: int
    emotion_name: str
    confidence: float
    probabilities: Dict[int, float]


class BatchEmotionResponse(BaseModel):
    results: List[EmotionResponse]


# 모델 로드 함수
def load_model():
    global model, tokenizer, device
    
    print("모델 로딩 중...")
    
    # 환경변수에서 모델 경로 불러오기
    model_path = os.getenv("MODEL_PATH", "./kobert_emotion_weighted")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        "monologg/kobert",
        trust_remote_code=True
    )
    
    # 모델 로드
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"✓ 모델 로드 완료 (Device: {device})")


# 예측 함수
def predict_emotion(text: str) -> Dict:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다.")
    
    # 토크나이징
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 예측
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(logits, dim=1).item()
        confidence = probs[0][pred_label].item()
    
    # 확률 딕셔너리 생성
    probabilities = {i: float(probs[0][i]) for i in range(6)}
    
    return {
        "text": text,
        "predicted_label": pred_label + 1,  # 1~6으로 변환
        "emotion_name": EMOTION_LABELS[pred_label],
        "confidence": confidence,
        "probabilities": {k+1: v for k, v in probabilities.items()}  # 1~6으로 변환
    }


# 시작 이벤트: 모델 로드
@app.on_event("startup")
async def startup_event():
    load_model()


# 루트 엔드포인트
@app.get("/textEmotion/")
async def root():
    return {
        "message": "감정 분류 API에 오신 것을 환영합니다!",
        "endpoints": {
            "POST /textEmotion/predict": "단일 텍스트 감정 분류",
            "POST /textEmotion/predict/batch": "여러 텍스트 일괄 분류",
            "GET /textEmotion/health": "서버 상태 확인"
        }
    }


# 헬스체크 엔드포인트
@app.get("/textEmotion/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }


# 단일 텍스트 예측 엔드포인트
@app.post("/textEmotion/predict", response_model=EmotionResponse)
async def predict_single(request: TextRequest):
    """
    단일 텍스트의 감정을 분류합니다.
    
    - **text**: 분석할 텍스트
    
    Returns:
    - **predicted_label**: 예측된 라벨 (1~6)
    - **emotion_name**: 감정 이름
    - **confidence**: 예측 확신도 (0~1)
    - **probabilities**: 각 라벨별 확률
    """
    try:
        if not request.text or request.text.strip() == "":
            raise HTTPException(status_code=400, detail="텍스트가 비어있습니다.")
        
        result = predict_emotion(request.text)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")


# 배치 예측 엔드포인트
@app.post("/textEmotion/predict/batch", response_model=BatchEmotionResponse)
async def predict_batch(request: BatchTextRequest):
    """
    여러 텍스트의 감정을 일괄 분류합니다.
    
    - **texts**: 분석할 텍스트 리스트
    
    Returns:
    - **results**: 각 텍스트의 예측 결과 리스트
    """
    try:
        if not request.texts or len(request.texts) == 0:
            raise HTTPException(status_code=400, detail="텍스트 리스트가 비어있습니다.")
        
        if len(request.texts) > 100:
            raise HTTPException(status_code=400, detail="한 번에 최대 100개까지만 처리 가능합니다.")
        
        results = []
        for text in request.texts:
            if text and text.strip():
                result = predict_emotion(text)
                results.append(result)
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"배치 예측 중 오류 발생: {str(e)}")


# 서버 실행 (로컬 전용)
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # main.py 파일의 app 객체
        host="127.0.0.1",  # 로컬호스트만 허용
        port=8000,
        reload=True  # 개발 시 자동 재시작
    )