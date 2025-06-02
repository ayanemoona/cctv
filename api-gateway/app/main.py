# api-gateway/main.py (핵심 부분)
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import httpx
import logging
from typing import List, Dict, Any
import asyncio
from datetime import datetime
import uuid

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Police Investigation CCTV Analysis System",
    description="경찰청 수사용 CCTV 분석 시스템",
    version="3.1.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 엔드포인트 설정
SERVICES = {
    "yolo": "http://localhost:8001",
    "clothing": "http://localhost:8002",
    "video": "http://localhost:8004"
}

# 수사 데이터베이스
investigation_cases = {}
suspect_database = {}

async def check_service_health(service_name: str, url: str) -> Dict[str, Any]:
    """개별 서비스 상태 확인"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")
            if response.status_code == 200:
                return {"service": service_name, "status": "healthy", "url": url}
            else:
                return {"service": service_name, "status": "unhealthy", "url": url}
    except Exception as e:
        return {"service": service_name, "status": "unreachable", "url": url, "error": str(e)}

@app.get("/")
async def root():
    return {
        "service": "Police Investigation CCTV Analysis System",
        "version": "3.1.0",
        "description": "경찰청 수사용 CCTV 분석 시스템 (크롭 이미지 지원)",
        "features": [
            "AI 기반 용의자 자동 탐지",
            "CCTV 영상 실시간 분석",
            "용의자 크롭 이미지 자동 생성",
            "웹 기반 증거 이미지 확인"
        ],
        "endpoints": {
            "dashboard": "/police/dashboard",
            "image_viewer": "/police/image_viewer"
        }
    }

@app.get("/health")
async def health_check():
    """전체 시스템 상태 확인"""
    core_services = ["yolo", "clothing", "video"]
    
    health_checks = [
        check_service_health(name, SERVICES[name]) 
        for name in core_services
    ]
    
    results = await asyncio.gather(*health_checks, return_exceptions=True)
    
    healthy_services = []
    for result in results:
        if isinstance(result, dict) and result["status"] == "healthy":
            healthy_services.append(result)
    
    core_healthy = len(healthy_services) >= 3
    
    return {
        "overall_status": "operational" if core_healthy else "critical",
        "system_type": "police_investigation_system",
        "healthy_count": len(healthy_services),
        "active_cases": len(investigation_cases),
        "registered_suspects": len(suspect_database)
    }

@app.post("/police/register_suspect")
async def police_register_suspect(
    suspect_id: str = Form(...),
    suspect_name: str = Form("미상"),
    case_number: str = Form(""),
    officer_name: str = Form(""),
    clothing_image: UploadFile = File(...)
):
    """용의자 등록"""
    try:
        logger.info(f"🚨 용의자 등록: {suspect_id}")
        
        if not clothing_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
        
        # Hybrid Clothing 서비스 호출
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": (clothing_image.filename, await clothing_image.read(), clothing_image.content_type)}
            data = {"person_id": suspect_id}
            
            response = await client.post(
                f"{SERVICES['clothing']}/register_person",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 수사 데이터베이스에 저장
                suspect_database[suspect_id] = {
                    "suspect_id": suspect_id,
                    "suspect_name": suspect_name,
                    "case_number": case_number,
                    "officer_name": officer_name,
                    "registration_time": datetime.now().isoformat(),
                    "status": "active"
                }
                
                logger.info(f"✅ 용의자 등록 완료: {suspect_id}")
                
                return {
                    "status": "success",
                    "message": f"용의자 '{suspect_id}' 등록 완료",
                    "suspect_info": {
                        "suspect_id": suspect_id,
                        "suspect_name": suspect_name,
                        "officer_name": officer_name
                    }
                }
            else:
                raise HTTPException(status_code=response.status_code, detail="AI 분석 시스템 오류")
                
    except Exception as e:
        logger.error(f"❌ 용의자 등록 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"용의자 등록 실패: {str(e)}")

@app.post("/police/analyze_cctv")
async def police_analyze_cctv(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    location: str = Form(...),
    date: str = Form(...),
    officer_name: str = Form(""),
    case_number: str = Form(""),
    fps_interval: float = Form(3.0),
    stop_on_detect: bool = Form(True)
):
    """CCTV 영상 분석"""
    try:
        if not video_file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="비디오 파일만 업로드 가능합니다")
        
        # 수사 케이스 ID 생성
        case_id = f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"🎬 CCTV 분석 시작: {case_id}")
        
        # 영상 내용 읽기
        content = await video_file.read()
        
        # 수사 케이스 정보 저장
        investigation_cases[case_id] = {
            "case_id": case_id,
            "case_number": case_number,
            "location": location,
            "date": date,
            "officer_name": officer_name,
            "video_filename": video_file.filename,
            "fps_interval": fps_interval,
            "stop_on_detect": stop_on_detect,
            "start_time": datetime.now().isoformat(),
            "status": "analyzing"
        }
        
        # 비디오 분석 서비스 호출
        analysis_endpoint = "/analyze_video_realtime" if stop_on_detect else "/analyze_video"
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            files = {"video_file": (video_file.filename, content, video_file.content_type)}
            data = {
                "fps_interval": fps_interval,
                "location": location,
                "date": date,
                "stop_on_detect": stop_on_detect
            }
            
            response = await client.post(
                f"{SERVICES['video']}{analysis_endpoint}",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_id = result.get('analysis_id')
                
                # 수사 케이스에 분석 ID 연결
                investigation_cases[case_id]["analysis_id"] = analysis_id
                
                logger.info(f"✅ CCTV 분석 시작 완료: {case_id}")
                
                return {
                    "status": "analysis_started",
                    "case_id": case_id,
                    "analysis_id": analysis_id,
                    "realtime_mode": stop_on_detect,
                    "message": f"케이스 '{case_id}' 분석이 시작되었습니다",
                    "monitoring": {
                        "status_check": f"/police/case_status/{case_id}",
                        "results": f"/police/case_report/{case_id}",
                        "image_viewer": f"/police/image_viewer/{case_id}"
                    }
                }
            else:
                raise HTTPException(status_code=response.status_code, detail="CCTV 분석 시스템 오류")
                
    except Exception as e:
        logger.error(f"❌ CCTV 분석 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CCTV 분석 실패: {str(e)}")

@app.get("/police/case_status/{case_id}")
async def get_police_case_status(case_id: str):
    """수사 케이스 진행 상황 조회"""
    if case_id not in investigation_cases:
        raise HTTPException(status_code=404, detail="수사 케이스를 찾을 수 없습니다")
    
    case_info = investigation_cases[case_id]
    analysis_id = case_info.get("analysis_id")
    
    if not analysis_id:
        return {"status": "preparing", "case_info": case_info}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{SERVICES['video']}/analysis_status/{analysis_id}")
            
            if response.status_code == 200:
                analysis_status = response.json()
                
                return {
                    "case_id": case_id,
                    "case_info": case_info,
                    "analysis_status": analysis_status,
                    "investigation_progress": {
                        "suspects_found": analysis_status.get("suspects_found", 0),
                        "crop_images_available": analysis_status.get("crop_images_available", 0),
                        "progress_percentage": analysis_status.get("progress", 0)
                    }
                }
            else:
                raise HTTPException(status_code=response.status_code, detail="분석 상태 조회 실패")
                
    except Exception as e:
        logger.error(f"❌ 케이스 상태 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"상태 조회 실패: {str(e)}")

@app.get("/police/case_report/{case_id}")
async def get_police_case_report(case_id: str):
    """경찰 수사 보고서 생성"""
    if case_id not in investigation_cases:
        raise HTTPException(status_code=404, detail="수사 케이스를 찾을 수 없습니다")
    
    case_info = investigation_cases[case_id]
    analysis_id = case_info.get("analysis_id")
    
    if not analysis_id:
        raise HTTPException(status_code=400, detail="분석이 시작되지 않았습니다")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{SERVICES['video']}/analysis_result/{analysis_id}")
            
            if response.status_code == 200:
                analysis_result = response.json()
                
                analysis_status = analysis_result.get("status")
                if analysis_status not in ["completed", "completed_early"]:
                    return {
                        "case_id": case_id,
                        "status": "incomplete",
                        "message": "분석이 아직 완료되지 않았습니다",
                        "current_progress": analysis_result.get("progress", 0),
                        "case_info": case_info
                    }
                
                # 간단한 보고서 생성
                suspects_timeline = analysis_result.get("suspects_timeline", [])
                crop_images = analysis_result.get("suspect_crop_images", [])
                
                report = {
                    "investigation_summary": {
                        "case_id": case_info["case_id"],
                        "location": case_info["location"],
                        "date": case_info["date"],
                        "officer_name": case_info.get("officer_name", ""),
                        "analysis_completion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    "detection_results": {
                        "total_suspects_detected": len(set(s["suspect_id"] for s in suspects_timeline)),
                        "total_evidence_images": len(crop_images),
                        "analysis_frames": analysis_result.get("total_frames_analyzed", 0)
                    },
                    "evidence_package": {
                        "cropped_suspect_images": crop_images,
                        "timeline_data": suspects_timeline
                    }
                }
                
                # 수사 케이스 완료 처리
                investigation_cases[case_id]["status"] = "completed"
                investigation_cases[case_id]["completion_time"] = datetime.now().isoformat()
                
                return report
                
            elif response.status_code == 400:
                return {
                    "case_id": case_id,
                    "status": "incomplete", 
                    "message": "분석이 진행 중입니다",
                    "case_info": case_info
                }
            else:
                raise HTTPException(status_code=response.status_code, detail="분석 결과 조회 실패")
                
    except Exception as e:
        logger.error(f"❌ 수사 보고서 생성 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"보고서 생성 실패: {str(e)}")

@app.get("/police/all_cases")
async def get_all_investigation_cases():
    """모든 수사 케이스 목록 조회"""
    return {
        "total_cases": len(investigation_cases),
        "cases": [
            {
                "case_id": case_id,
                "location": info.get("location", ""),
                "date": info.get("date", ""),
                "officer_name": info.get("officer_name", ""),
                "status": info.get("status", "unknown")
            }
            for case_id, info in investigation_cases.items()
        ]
    }

# 기존 호환성 엔드포인트
@app.post("/register_suspect")
async def register_suspect(
    suspect_id: str = Form(...),
    suspect_name: str = Form(""),
    case_number: str = Form(""),
    clothing_image: UploadFile = File(...)
):
    """기존 용의자 등록 엔드포인트"""
    return await police_register_suspect(
        suspect_id=suspect_id,
        suspect_name=suspect_name,
        case_number=case_number,
        clothing_image=clothing_image
    )

@app.post("/analyze_video")
async def analyze_video(
    video_file: UploadFile = File(...),
    fps_interval: float = Form(5.0),
    location: str = Form(""),
    date: str = Form("")
):
    """기존 영상 분석 엔드포인트"""
    try:
        if not video_file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="비디오 파일만 업로드 가능합니다")
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            files = {"video_file": (video_file.filename, await video_file.read(), video_file.content_type)}
            data = {
                "fps_interval": fps_interval,
                "location": location,
                "date": date,
                "stop_on_detect": False
            }
            
            response = await client.post(
                f"{SERVICES['video']}/analyze_video",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail="영상 분석 서비스 오류")
                
    except Exception as e:
        logger.error(f"❌ 영상 분석 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"영상 분석 실패: {str(e)}")

try:
    from web_routes import router as web_router
    app.include_router(web_router)
    logger.info("✅ 웹 라우터 로드 완료")
except ImportError:
    logger.warning("⚠️ web_routes.py 파일이 없습니다. 웹 페이지를 사용하려면 web_routes.py를 생성하세요.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)