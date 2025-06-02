# hybrid_matcher.py
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import json
import base64
import io
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class HybridClothingMatcher:
    """
    Computer Vision + MobileNet 결합한 옷차림 매칭 시스템
    경량화되고 배포 안정적인 솔루션
    """
    
    def __init__(self):
        self.registered_suspects = {}
        self.setup_mobilenet()
        logger.info("🎯 Hybrid Clothing Matcher 초기화 완료")
        
    def setup_mobilenet(self):
        """MobileNet 모델 초기화"""
        try:
            # MobileNetV3 Small (가벼운 모델)
            self.mobilenet = models.mobilenet_v3_small(pretrained=True)
            # 특징 추출기로 사용 (분류층 제거)
            self.mobilenet.classifier = torch.nn.Identity()
            self.mobilenet.eval()
            
            # 이미지 전처리
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("✅ MobileNet 모델 로드 완료 (~100MB)")
            
        except Exception as e:
            logger.error(f"❌ MobileNet 로드 실패: {e}")
            self.mobilenet = None
    
    def extract_color_features(self, image):
        """색상 특징 추출 (Computer Vision)"""
        # RGB 히스토그램
        color_features = []
        for i in range(3):  # R, G, B
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            color_features.extend(hist.flatten())
        
        # HSV 히스토그램 (색상 매칭에 더 효과적)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180])  # 색조
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])  # 채도
        
        color_features.extend(h_hist.flatten())
        color_features.extend(s_hist.flatten())
        
        return np.array(color_features)
    
    def extract_texture_features(self, image):
        """텍스처/패턴 특징 추출 (Computer Vision)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. LBP (Local Binary Pattern) - 텍스처 분석
        lbp = self.calculate_lbp(gray)
        
        # 2. 엣지 히스토그램 - 패턴 분석
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
        
        # 3. 그래디언트 방향 히스토그램
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_hist = cv2.calcHist([grad_mag.astype(np.uint8)], [0], None, [16], [0, 256])
        
        texture_features = np.concatenate([
            lbp.flatten(),
            edge_hist.flatten(), 
            grad_hist.flatten()
        ])
        
        return texture_features
    
    def calculate_lbp(self, gray_image, radius=1, n_points=8):
        """Local Binary Pattern 계산"""
        h, w = gray_image.shape
        lbp_image = np.zeros_like(gray_image)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray_image[i, j]
                
                # 8방향 이웃 픽셀과 비교
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                # 이진 패턴 생성
                binary_value = 0
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        binary_value += 2**k
                
                lbp_image[i, j] = binary_value
        
        # LBP 히스토그램
        lbp_hist = cv2.calcHist([lbp_image], [0], None, [32], [0, 256])
        return lbp_hist
    
    def extract_mobilenet_features(self, image):
        """MobileNet으로 고수준 특징 추출"""
        if self.mobilenet is None:
            return np.zeros(576)  # 기본 크기 반환
        
        try:
            # PIL Image로 변환
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # 전처리 및 추론
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                features = self.mobilenet(input_tensor)
                # L2 정규화
                features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"MobileNet 특징 추출 실패: {e}")
            return np.zeros(576)
    
    def extract_hybrid_features(self, image):
        """Hybrid 특징 추출: CV + MobileNet"""
        # 1. Computer Vision 특징
        color_features = self.extract_color_features(image)
        texture_features = self.extract_texture_features(image)
        cv_features = np.concatenate([color_features, texture_features])
        
        # 2. MobileNet 특징  
        mobilenet_features = self.extract_mobilenet_features(image)
        
        # 3. 특징 크기 정규화
        cv_size = 512
        mobilenet_size = 256
        
        if len(cv_features) > cv_size:
            cv_features = cv_features[:cv_size]
        elif len(cv_features) < cv_size:
            cv_features = np.pad(cv_features, (0, cv_size - len(cv_features)))
            
        if len(mobilenet_features) > mobilenet_size:
            mobilenet_features = mobilenet_features[:mobilenet_size]
        elif len(mobilenet_features) < mobilenet_size:
            mobilenet_features = np.pad(mobilenet_features, (0, mobilenet_size - len(mobilenet_features)))
        
        # 4. 가중 결합 (CV 70%, MobileNet 30%)
        cv_weight = 0.7
        mobilenet_weight = 0.3
        
        hybrid_features = np.concatenate([
            cv_features * cv_weight,
            mobilenet_features * mobilenet_weight
        ])
        
        # 5. L2 정규화
        hybrid_features = hybrid_features / (np.linalg.norm(hybrid_features) + 1e-8)
        
        return hybrid_features
    
    def register_suspect(self, suspect_id: str, clothing_image: np.ndarray) -> Dict[str, Any]:
        """용의자 옷차림 등록"""
        try:
            # 특징 추출
            features = self.extract_hybrid_features(clothing_image)
            
            # 등록 정보 저장
            self.registered_suspects[suspect_id] = {
                "features": features.tolist(),
                "feature_size": len(features),
                "registration_method": "hybrid_cv_mobilenet"
            }
            
            logger.info(f"✅ 용의자 등록 완료: {suspect_id}")
            
            return {
                "status": "success",
                "suspect_id": suspect_id,
                "feature_dimension": len(features),
                "method": "hybrid_cv_mobilenet",
                "message": f"용의자 '{suspect_id}' 옷차림이 등록되었습니다"
            }
            
        except Exception as e:
            logger.error(f"❌ 용의자 등록 실패: {e}")
            return {
                "status": "error",
                "message": f"등록 실패: {str(e)}"
            }
    
    def match_clothing(self, query_image: np.ndarray, threshold: float = 0.7) -> Dict[str, Any]:
        """CCTV 이미지와 등록된 옷차림 매칭"""
        try:
            if not self.registered_suspects:
                return {
                    "status": "no_suspects",
                    "message": "등록된 용의자가 없습니다",
                    "matches": []
                }
            
            # 쿼리 이미지 특징 추출
            query_features = self.extract_hybrid_features(query_image)
            
            matches = []
            for suspect_id, suspect_data in self.registered_suspects.items():
                stored_features = np.array(suspect_data["features"])
                
                # 코사인 유사도 계산
                similarity = cosine_similarity(
                    query_features.reshape(1, -1),
                    stored_features.reshape(1, -1)
                )[0][0]
                
                if similarity >= threshold:
                    confidence = "high" if similarity > 0.8 else "medium"
                    
                    matches.append({
                        "suspect_id": suspect_id,
                        "similarity": float(similarity),
                        "confidence": confidence,
                        "similarity_percentage": float(similarity * 100),
                        "match": True
                    })
            
            # 유사도 순으로 정렬
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.info(f"매칭 완료: {len(matches)}명 발견")
            
            return {
                "status": "success",
                "total_comparisons": len(self.registered_suspects),
                "matches_found": len(matches),
                "threshold": threshold,
                "matches": matches,
                "method": "hybrid_cv_mobilenet"
            }
            
        except Exception as e:
            logger.error(f"❌ 매칭 실패: {e}")
            return {
                "status": "error",
                "message": f"매칭 실패: {str(e)}",
                "matches": []
            }
    
    def get_registered_suspects(self) -> Dict[str, Any]:
        """등록된 용의자 목록 조회"""
        return {
            "status": "success",
            "total_suspects": len(self.registered_suspects),
            "suspect_ids": list(self.registered_suspects.keys()),
            "method": "hybrid_cv_mobilenet"
        }
    
    def delete_suspect(self, suspect_id: str) -> Dict[str, Any]:
        """용의자 삭제"""
        if suspect_id in self.registered_suspects:
            del self.registered_suspects[suspect_id]
            return {
                "status": "success",
                "message": f"용의자 '{suspect_id}'가 삭제되었습니다",
                "remaining_suspects": len(self.registered_suspects)
            }
        else:
            return {
                "status": "error",
                "message": f"용의자 '{suspect_id}'를 찾을 수 없습니다"
            }
    
    def analyze_clothing_details(self, image: np.ndarray) -> Dict[str, Any]:
        """옷차림 상세 분석 (디버깅/시연용)"""
        try:
            # 색상 분석
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 주요 색상 추출
            dominant_colors = self.get_dominant_colors(image)
            
            # 텍스처 분석
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 패턴 분석
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return {
                "dominant_colors": dominant_colors,
                "texture_complexity": "high" if texture_variance > 500 else "medium" if texture_variance > 100 else "low",
                "pattern_density": "high" if edge_density > 0.1 else "medium" if edge_density > 0.05 else "low",
                "analysis_method": "hybrid_cv_analysis"
            }
            
        except Exception as e:
            logger.error(f"옷차림 분석 실패: {e}")
            return {"error": str(e)}
    
    def get_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[str]:
        """주요 색상 추출"""
        try:
            # 이미지를 1차원으로 변환
            data = image.reshape((-1, 3))
            data = np.float32(data)
            
            # K-means 클러스터링
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 색상을 문자열로 변환
            colors = []
            for center in centers:
                b, g, r = center.astype(int)
                color_name = self.rgb_to_color_name(r, g, b)
                colors.append(color_name)
            
            return colors
            
        except Exception as e:
            return ["알 수 없음"]
    
    def rgb_to_color_name(self, r: int, g: int, b: int) -> str:
        """RGB 값을 색상 이름으로 변환"""
        # 간단한 색상 분류
        if r > 200 and g > 200 and b > 200:
            return "흰색"
        elif r < 50 and g < 50 and b < 50:
            return "검은색"
        elif r > 150 and g < 100 and b < 100:
            return "빨간색"
        elif r < 100 and g > 150 and b < 100:
            return "초록색"
        elif r < 100 and g < 100 and b > 150:
            return "파란색"
        elif r > 150 and g > 150 and b < 100:
            return "노란색"
        elif r > 150 and g < 100 and b > 150:
            return "보라색"
        elif r > 100 and g > 100 and b > 100:
            return "회색"
        else:
            return "기타색상"

# 테스트 코드
if __name__ == "__main__":
    matcher = HybridClothingMatcher()
    print("✅ Hybrid Clothing Matcher 테스트 완료")