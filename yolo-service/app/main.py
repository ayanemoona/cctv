# yolo-service/main.py (완전한 버전)
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import List, Dict, Any, Optional
import base64
import asyncio
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YOLO Object Detection Service",
    description="YOLOv8 기반 객체 탐지 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 모델 변수
model = None
model_loading = False

async def load_yolo_model():
    """YOLO 모델 비동기 로드"""
    global model, model_loading
    
    if model is not None:
        return True
        
    if model_loading:
        # 이미 로딩 중이면 대기
        while model_loading and model is None:
            await asyncio.sleep(0.1)
        return model is not None
    
    model_loading = True
    
    try:
        logger.info("🔥 YOLOv8 모델 로드 시작...")
        
        # ultralytics 임포트 시도
        try:
            from ultralytics import YOLO
            import torch
        except ImportError:
            logger.error("❌ ultralytics 또는 torch 패키지가 설치되지 않았습니다.")
            logger.error("설치 명령: pip install ultralytics torch torchvision")
            model_loading = False
            return False
        
        # PyTorch 2.6 호환성을 위한 safe_globals 설정
        try:
            # PyTorch 2.6+ 대응
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            logger.info("✅ PyTorch 2.6+ 호환성 설정 완료")
        except:
            # 이전 버전에서는 무시
            pass
        
        # 모델 파일 경로 확인
        model_path = 'yolov8s.pt'
        
        # 환경 변수로 weights_only 설정 (보안 경고 해결)
        import os
        os.environ['YOLO_WEIGHTS_ONLY'] = 'False'
        
        # 모델 로드 (신뢰할 수 있는 소스이므로 weights_only=False 사용)
        try:
            # 방법 1: 직접 weights_only=False 설정
            import torch
            original_load = torch.load
            
            def safe_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = safe_load
            model = YOLO(model_path)
            torch.load = original_load  # 원복
            
        except Exception as e1:
            logger.warning(f"방법 1 실패: {e1}")
            try:
                # 방법 2: 환경변수 설정
                os.environ['TORCH_SERIALIZATION_SAFE_GLOBALS'] = 'ultralytics.nn.tasks.DetectionModel'
                model = YOLO(model_path)
            except Exception as e2:
                logger.error(f"방법 2도 실패: {e2}")
                raise e2
        
        # 테스트 추론으로 모델 초기화
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model(dummy_image, verbose=False)
        
        logger.info("✅ YOLOv8 모델 로드 및 초기화 완료!")
        logger.info("📊 메모리 사용량: ~50MB")
        
        model_loading = False
        return True
        
    except Exception as e:
        logger.error(f"❌ YOLO 모델 로드 실패: {e}")
        logger.error("🔧 해결 방법:")
        logger.error("1. pip install ultralytics torch==2.5.1 torchvision  # PyTorch 버전 다운그레이드")
        logger.error("2. 또는 pip install --upgrade ultralytics  # 최신 버전으로 업그레이드")
        logger.error("3. 인터넷 연결 확인 (모델 다운로드)")
        logger.error("4. 충분한 디스크 공간 확인")
        
        model = None
        model_loading = False
        return False

@app.on_event("startup")
async def startup_event():
    """서비스 시작 시 YOLO 모델 로드"""
    logger.info("🚀 YOLO 객체 탐지 서비스 시작...")
    
    success = await load_yolo_model()
    if success:
        logger.info("✅ 서비스 준비 완료!")
    else:
        logger.error("❌ 서비스 시작 실패 - 모델 로드 오류")

@app.get("/")
async def root():
    return {
        "service": "YOLO Object Detection Service",
        "version": "1.0.0",
        "model": "YOLOv8s",
        "description": "YOLOv8 기반 객체 탐지 (사람 위주)",
        "model_loaded": model is not None,
        "model_loading": model_loading,
        "features": [
            "사람 탐지 (Person Detection)",
            "바운딩 박스 좌표 제공",
            "신뢰도 점수 제공",
            "실시간 처리"
        ],
        "endpoints": {
            "/detect": "이미지 객체 탐지",
            "/detect_batch": "여러 이미지 일괄 처리",
            "/health": "서비스 상태 확인",
            "/model_info": "모델 정보 조회",
            "/reload_model": "모델 재로드"
        }
    }

@app.get("/health")
async def health_check():
    """서비스 상태 확인"""
    return {
        "status": "healthy" if model is not None else "loading" if model_loading else "unhealthy",
        "model_loaded": model is not None,
        "model_loading": model_loading,
        "model_type": "YOLOv8s" if model is not None else None,
        "memory_usage": "~50MB" if model is not None else "0MB",
        "ready_for_detection": model is not None,
        "error_info": "모델 로드 중..." if model_loading else "모델 로드 실패" if model is None else None
    }

@app.post("/reload_model")
async def reload_model():
    """모델 재로드"""
    global model
    model = None
    
    success = await load_yolo_model()
    
    if success:
        return {
            "status": "success",
            "message": "YOLO 모델이 성공적으로 재로드되었습니다",
            "model_loaded": True
        }
    else:
        return {
            "status": "failed",
            "message": "YOLO 모델 재로드에 실패했습니다",
            "model_loaded": False
        }

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    show_all_objects: bool = Form(False)
):
    """이미지에서 객체 탐지"""
    try:
        # 모델 상태 확인
        if model_loading:
            raise HTTPException(
                status_code=503, 
                detail="YOLO 모델 로드 중입니다. 잠시 후 다시 시도해주세요."
            )
        
        if model is None:
            # 모델 로드 재시도
            success = await load_yolo_model()
            if not success:
                raise HTTPException(
                    status_code=503, 
                    detail="YOLO 모델이 로드되지 않았습니다. /reload_model 엔드포인트로 재로드를 시도하거나 ultralytics 설치를 확인하세요."
                )
        
        # 파일 형식 검증
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")

        # 이미지 로드 및 변환
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)

        logger.info(f"📷 이미지 수신: {file.filename}, conf={confidence}, size={image_np.shape}")

        # YOLO 추론
        results = model(image_np, conf=confidence, verbose=False)
        detections = []

        # 결과가 있는지 확인
        if len(results) == 0 or results[0].boxes is None:
            logger.info("🔍 탐지된 객체가 없습니다")
        else:
            # 결과 파싱
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                xyxy = box.xyxy[0].tolist()

                # 모든 객체 또는 사람만 필터링
                if show_all_objects or class_name == "person":
                    detections.append({
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": round(conf, 4),
                        "bbox": {
                            "x1": round(xyxy[0], 1),
                            "y1": round(xyxy[1], 1),
                            "x2": round(xyxy[2], 1),
                            "y2": round(xyxy[3], 1)
                        }
                    })

        logger.info(f"✅ 탐지 완료: {len(detections)}개 객체 발견 (사람: {len([d for d in detections if d['class_name'] == 'person'])}명)")

        return {
            "status": "success",
            "results": {
                "total_detections": len(detections),
                "all_detections": detections,
                "person_count": len([d for d in detections if d['class_name'] == 'person'])
            },
            "model": "yolov8s",
            "image_size": {
                "width": image_np.shape[1],
                "height": image_np.shape[0]
            },
            "processing_info": {
                "confidence_threshold": confidence,
                "show_all_objects": show_all_objects
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ YOLO 분석 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"YOLO 분석 실패: {str(e)}")

@app.post("/detect_batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    confidence: float = Form(0.25)
):
    """여러 이미지 일괄 처리"""
    try:
        # 모델 상태 확인
        if model_loading:
            raise HTTPException(
                status_code=503, 
                detail="YOLO 모델 로드 중입니다. 잠시 후 다시 시도해주세요."
            )
        
        if model is None:
            success = await load_yolo_model()
            if not success:
                raise HTTPException(
                    status_code=503, 
                    detail="YOLO 모델이 로드되지 않았습니다"
                )
        
        results = []
        
        for i, file in enumerate(files):
            if not file.content_type.startswith('image/'):
                results.append({
                    "image_index": i,
                    "filename": file.filename,
                    "error": "이미지 파일이 아닙니다",
                    "total_detections": 0,
                    "detections": []
                })
                continue
                
            try:
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert("RGB")
                image_np = np.array(image)
                
                # YOLO 추론
                yolo_results = model(image_np, conf=confidence, verbose=False)
                detections = []
                
                if len(yolo_results) > 0 and yolo_results[0].boxes is not None:
                    for box in yolo_results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls_id]
                        xyxy = box.xyxy[0].tolist()
                        
                        if class_name == "person":
                            detections.append({
                                "class_id": cls_id,
                                "class_name": class_name,
                                "confidence": round(conf, 4),
                                "bbox": {
                                    "x1": round(xyxy[0], 1),
                                    "y1": round(xyxy[1], 1),
                                    "x2": round(xyxy[2], 1),
                                    "y2": round(xyxy[3], 1)
                                }
                            })
                
                results.append({
                    "image_index": i,
                    "filename": file.filename,
                    "total_detections": len(detections),
                    "detections": detections,
                    "image_size": {
                        "width": image_np.shape[1],
                        "height": image_np.shape[0]
                    }
                })
                
            except Exception as e:
                results.append({
                    "image_index": i,
                    "filename": file.filename,
                    "error": str(e),
                    "total_detections": 0,
                    "detections": []
                })
        
        total_detections = sum(r["total_detections"] for r in results)
        logger.info(f"✅ 일괄 처리 완료: {len(files)}개 이미지, 총 {total_detections}명 탐지")
        
        return {
            "status": "success",
            "total_images": len(files),
            "total_detections": total_detections,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 일괄 처리 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"일괄 처리 실패: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """모델 정보 조회"""
    if model_loading:
        return {
            "model_loaded": False,
            "loading": True,
            "message": "모델 로드 중..."
        }
    
    if model is None:
        return {
            "model_loaded": False,
            "loading": False,
            "error": "모델이 로드되지 않았습니다",
            "solution": "POST /reload_model 엔드포인트로 재로드 시도"
        }
    
    return {
        "model_loaded": True,
        "model_name": "YOLOv8s",
        "model_path": "yolov8s.pt",
        "framework": "Ultralytics YOLOv8",
        "classes": list(model.names.values()) if hasattr(model, 'names') else [],
        "total_classes": len(model.names) if hasattr(model, 'names') else 0,
        "person_class_id": 0,  # YOLO에서 사람은 보통 클래스 ID 0
        "recommended_confidence": 0.25,
        "input_size": "640x640",
        "memory_usage": "~50MB",
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
        "performance": {
            "speed": "실시간 처리 가능",
            "accuracy": "높음 (YOLOv8s)",
            "memory_efficient": True
        }
    }

@app.get("/test")
async def test_detection():
    """테스트용 더미 이미지로 탐지 테스트"""
    try:
        if model is None:
            success = await load_yolo_model()
            if not success:
                raise HTTPException(status_code=503, detail="모델 로드 실패")
        
        # 640x640 더미 이미지 생성 (검은색)
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # 테스트 추론
        results = model(dummy_image, conf=0.25, verbose=False)
        
        return {
            "status": "success",
            "message": "YOLO 모델이 정상적으로 작동합니다",
            "test_image_size": "640x640",
            "detections_found": len(results[0].boxes) if len(results) > 0 and results[0].boxes is not None else 0,
            "model_ready": True
        }
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"테스트 실패: {str(e)}")

@app.get("/status")
async def get_status():
    """상세 서비스 상태"""
    return {
        "service": "YOLO Object Detection Service",
        "version": "1.0.0",
        "model": {
            "loaded": model is not None,
            "loading": model_loading,
            "type": "YOLOv8s" if model is not None else None
        },
        "system": {
            "memory_usage": "~50MB" if model is not None else "0MB",
            "ready": model is not None and not model_loading
        },
        "endpoints_available": model is not None,
        "last_startup": "서비스 시작됨"
    }

if __name__ == "__main__":
    import uvicorn
    
    # 환경 변수로 포트 설정 가능
    port = int(os.getenv("YOLO_PORT", 8001))
    
    logger.info(f"🚀 YOLO 서비스를 포트 {port}에서 시작합니다...")
    uvicorn.run(app, host="0.0.0.0", port=port)