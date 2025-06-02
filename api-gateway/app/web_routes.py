# api-gateway/web_routes.py (수정된 버전)
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/police/dashboard", response_class=HTMLResponse)
async def police_dashboard():
    """경찰 수사관 전용 대시보드"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>경찰청 CCTV 수사 분석 시스템</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%);
                min-height: 100vh;
                color: #fff;
            }
            .header {
                background: rgba(0, 0, 0, 0.3);
                padding: 20px 0;
                border-bottom: 3px solid #ffd700;
                backdrop-filter: blur(10px);
            }
            .header-content {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 20px;
            }
            .logo h1 {
                font-size: 1.8em;
                color: #ffd700;
                font-weight: bold;
            }
            .badge {
                background: #dc3545;
                color: white;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 30px 20px; }
            
            .main-sections {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 30px;
            }
            .section-card {
                background: rgba(255, 255, 255, 0.15);
                border-radius: 20px;
                padding: 30px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: all 0.3s ease;
            }
            .section-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
            }
            .suspect-registration { border-left: 5px solid #dc3545; }
            .video-analysis { border-left: 5px solid #007bff; }
            
            .section-header {
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 25px;
            }
            .section-icon { font-size: 2.5em; }
            .section-title { font-size: 1.5em; font-weight: bold; }
            
            .upload-area {
                background: rgba(255, 255, 255, 0.1);
                border: 2px dashed rgba(255, 255, 255, 0.3);
                border-radius: 15px;
                padding: 40px 20px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 20px 0;
            }
            .upload-area:hover {
                border-color: #ffd700;
                background: rgba(255, 215, 0, 0.1);
            }
            .upload-icon { font-size: 3em; margin-bottom: 15px; opacity: 0.7; }
            
            .form-group { margin-bottom: 20px; }
            .form-label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #ffd700;
            }
            .form-input {
                width: 100%;
                padding: 12px 15px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                font-size: 16px;
            }
            .form-input::placeholder { color: rgba(255, 255, 255, 0.6); }
            .form-input:focus {
                outline: none;
                border-color: #ffd700;
                background: rgba(255, 255, 255, 0.15);
            }
            
            .btn {
                padding: 15px 30px;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .btn-primary {
                background: linear-gradient(45deg, #dc3545, #c82333);
                color: white;
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(220, 53, 69, 0.4);
            }
            .btn-secondary {
                background: linear-gradient(45deg, #007bff, #0056b3);
                color: white;
            }
            .btn-secondary:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0, 123, 255, 0.4);
            }
            
            .alert {
                padding: 15px 20px;
                border-radius: 10px;
                margin: 15px 0;
                border-left: 5px solid;
                display: none;
            }
            .alert-success {
                background: rgba(40, 167, 69, 0.2);
                border-left-color: #28a745;
                color: #d4edda;
            }
            .alert-error {
                background: rgba(220, 53, 69, 0.2);
                border-left-color: #dc3545;
                color: #f8d7da;
            }
            
            .uploaded-media {
                max-width: 100%;
                border-radius: 10px;
                margin: 15px 0;
                border: 3px solid rgba(255, 255, 255, 0.3);
            }
            
            .quick-actions {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 20px;
                margin-top: 30px;
                text-align: center;
            }
            .quick-actions h3 {
                margin-bottom: 20px;
                color: #ffd700;
            }
            .quick-btn {
                display: inline-block;
                margin: 10px;
                padding: 15px 25px;
                background: linear-gradient(45deg, #28a745, #20903c);
                color: white;
                text-decoration: none;
                border-radius: 10px;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .quick-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(40, 167, 69, 0.4);
                color: white;
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-content">
                <div class="logo">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        <div style="font-size: 2em;">🚔</div>
                        <div>
                            <h1>경찰청 CCTV 수사 분석 시스템</h1>
                            <div style="font-size: 14px; opacity: 0.8;">AI 기반 용의자 추적 및 크롭 이미지 생성</div>
                        </div>
                    </div>
                </div>
                <div class="badge">RESTRICTED</div>
            </div>
        </div>

        <div class="container">
            <!-- 메인 섹션 -->
            <div class="main-sections">
                <!-- 용의자 등록 섹션 -->
                <div class="section-card suspect-registration">
                    <div class="section-header">
                        <div class="section-icon">🚨</div>
                        <div>
                            <div class="section-title">용의자 등록</div>
                            <div style="font-size: 14px; opacity: 0.8;">수배자 또는 추적 대상의 옷차림 이미지 등록</div>
                        </div>
                    </div>

                    <form id="suspectForm">
                        <div class="form-group">
                            <label class="form-label">용의자 ID *</label>
                            <input type="text" id="suspectId" class="form-input" placeholder="예: SUSPECT_001" required>
                        </div>

                        <div class="form-group">
                            <label class="form-label">담당 수사관</label>
                            <input type="text" id="officerName" class="form-input" placeholder="홍길동 경위">
                        </div>

                        <div class="upload-area" id="suspectUploadArea">
                            <div class="upload-icon">📷</div>
                            <div><strong>용의자 옷차림 이미지 업로드</strong></div>
                            <div style="font-size: 14px; opacity: 0.7; margin-top: 10px;">
                                전신 사진 권장 | JPG, PNG 지원 | 최대 10MB
                            </div>
                        </div>
                        <input type="file" id="suspectImageInput" accept="image/*" style="display: none;">

                        <div id="suspectPreview"></div>

                        <button type="submit" class="btn btn-primary" style="width: 100%;">
                            🚨 용의자 등록
                        </button>
                    </form>

                    <div class="alert alert-success" id="suspectAlert"></div>
                    <div class="alert alert-error" id="suspectError"></div>
                </div>

                <!-- CCTV 영상 분석 섹션 -->
                <div class="section-card video-analysis">
                    <div class="section-header">
                        <div class="section-icon">🎬</div>
                        <div>
                            <div class="section-title">CCTV 영상 분석</div>
                            <div style="font-size: 14px; opacity: 0.8;">영상에서 용의자 자동 탐지 및 크롭 이미지 생성</div>
                        </div>
                    </div>

                    <form id="videoForm">
                        <div class="form-group">
                            <label class="form-label">촬영 위치 *</label>
                            <input type="text" id="videoLocation" class="form-input" placeholder="예: 강남역 2번 출구" required>
                        </div>

                        <div class="form-group">
                            <label class="form-label">촬영 일시 *</label>
                            <input type="datetime-local" id="videoDate" class="form-input" required>
                        </div>

                        <div class="form-group">
                            <label class="form-label">담당 수사관 *</label>
                            <input type="text" id="videoOfficer" class="form-input" placeholder="홍길동 경위" required>
                        </div>

                        <div class="form-group">
                            <label class="form-label">분석 모드</label>
                            <select id="analysisMode" class="form-input">
                                <option value="true" selected>⚡ 실시간 모드 (용의자 발견 시 즉시 중단)</option>
                                <option value="false">📊 전체 분석 모드 (전체 영상 분석)</option>
                            </select>
                        </div>

                        <div class="upload-area" id="videoUploadArea">
                            <div class="upload-icon">🎥</div>
                            <div><strong>CCTV 영상 파일 업로드</strong></div>
                            <div style="font-size: 14px; opacity: 0.7; margin-top: 10px;">
                                MP4, AVI, MOV 지원 | 최대 500MB
                            </div>
                        </div>
                        <input type="file" id="videoFileInput" accept="video/*" style="display: none;">

                        <div id="videoPreview"></div>

                        <button type="submit" class="btn btn-secondary" style="width: 100%;">
                            🔍 CCTV 분석 시작
                        </button>
                    </form>

                    <div class="alert alert-success" id="videoAlert"></div>
                    <div class="alert alert-error" id="videoError"></div>
                </div>
            </div>
            
            <!-- 빠른 액션 -->
            <div class="quick-actions">
                <h3>🚀 빠른 액션</h3>
                <a href="/police/image_viewer" class="quick-btn">🖼️ 크롭 이미지 뷰어</a>
                <a href="/police/all_cases" class="quick-btn">📋 전체 케이스 조회</a>
            </div>
        </div>

        <script>
            const API_BASE = 'http://localhost:8000';

            document.addEventListener('DOMContentLoaded', function() {
                setupEventListeners();
            });

            function setupEventListeners() {
                // 용의자 이미지 업로드
                const suspectUploadArea = document.getElementById('suspectUploadArea');
                const suspectImageInput = document.getElementById('suspectImageInput');
                
                suspectUploadArea.addEventListener('click', () => suspectImageInput.click());
                suspectImageInput.addEventListener('change', handleSuspectFileSelect);

                // CCTV 영상 업로드
                const videoUploadArea = document.getElementById('videoUploadArea');
                const videoFileInput = document.getElementById('videoFileInput');
                
                videoUploadArea.addEventListener('click', () => videoFileInput.click());
                videoFileInput.addEventListener('change', handleVideoFileSelect);

                // 폼 제출
                document.getElementById('suspectForm').addEventListener('submit', handleSuspectSubmit);
                document.getElementById('videoForm').addEventListener('submit', handleVideoSubmit);
            }

            function handleSuspectFileSelect(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('suspectPreview').innerHTML = 
                            '<img src="' + e.target.result + '" class="uploaded-media" alt="용의자 이미지">';
                    };
                    reader.readAsDataURL(file);
                }
            }

            function handleVideoFileSelect(e) {
                const file = e.target.files[0];
                if (file) {
                    const url = URL.createObjectURL(file);
                    document.getElementById('videoPreview').innerHTML = 
                        '<video src="' + url + '" class="uploaded-media" controls style="max-height: 300px;"></video>';
                }
            }

            async function handleSuspectSubmit(e) {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('suspect_id', document.getElementById('suspectId').value);
                formData.append('officer_name', document.getElementById('officerName').value);
                formData.append('clothing_image', document.getElementById('suspectImageInput').files[0]);

                try {
                    const response = await fetch('/police/register_suspect', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    
                    if (response.ok) {
                        showAlert('suspectAlert', '✅ 용의자 등록이 완료되었습니다.');
                        document.getElementById('suspectForm').reset();
                        document.getElementById('suspectPreview').innerHTML = '';
                    } else {
                        showAlert('suspectError', '❌ 등록 실패: ' + result.detail);
                    }
                } catch (error) {
                    showAlert('suspectError', '❌ 등록 실패: ' + error.message);
                }
            }

            async function handleVideoSubmit(e) {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('video_file', document.getElementById('videoFileInput').files[0]);
                formData.append('location', document.getElementById('videoLocation').value);
                formData.append('date', document.getElementById('videoDate').value);
                formData.append('officer_name', document.getElementById('videoOfficer').value);
                formData.append('stop_on_detect', document.getElementById('analysisMode').value);

                try {
                    const response = await fetch('/police/analyze_cctv', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    
                    if (response.ok) {
                        const mode = result.realtime_mode ? '⚡ 실시간' : '📊 전체';
                        showAlert('videoAlert', `✅ ${mode} CCTV 분석이 시작되었습니다.\\n케이스 ID: ${result.case_id}\\n\\n🖼️ 이미지 확인: /police/image_viewer/${result.case_id}`);
                        
                        // 폼 리셋
                        document.getElementById('videoForm').reset();
                        document.getElementById('videoPreview').innerHTML = '';
                    } else {
                        showAlert('videoError', '❌ 분석 실패: ' + result.detail);
                    }
                } catch (error) {
                    showAlert('videoError', '❌ 분석 실패: ' + error.message);
                }
            }

            function showAlert(elementId, message) {
                const alertElement = document.getElementById(elementId);
                alertElement.textContent = message;
                alertElement.style.display = 'block';
                
                setTimeout(() => {
                    alertElement.style.display = 'none';
                }, 7000);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.get("/police/image_viewer/{case_id}", response_class=HTMLResponse)
async def police_image_viewer(case_id: str):
    """크롭 이미지 뷰어"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>크롭 이미지 뷰어 - {case_id}</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%);
                min-height: 100vh;
                color: #fff;
            }}
            .header {{
                background: rgba(0, 0, 0, 0.3);
                padding: 20px 0;
                border-bottom: 3px solid #ffd700;
                backdrop-filter: blur(10px);
            }}
            .header-content {{
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 20px;
            }}
            .logo h1 {{
                font-size: 1.8em;
                color: #ffd700;
                font-weight: bold;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; padding: 30px 20px; }}
            
            .case-info {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                text-align: center;
            }}
            
            .loading {{
                text-align: center;
                padding: 60px;
            }}
            .spinner {{
                border: 4px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top: 4px solid #ffd700;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            .images-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            
            .image-card {{
                background: rgba(255, 255, 255, 0.15);
                border-radius: 15px;
                padding: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: all 0.3s ease;
                text-align: center;
            }}
            
            .image-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
                border-color: #ffd700;
            }}
            
            .crop-image {{
                width: 100%;
                max-width: 200px;
                height: auto;
                border-radius: 10px;
                border: 2px solid rgba(255, 255, 255, 0.3);
                margin-bottom: 15px;
                cursor: pointer;
                transition: all 0.3s ease;
            }}
            
            .crop-image:hover {{
                border-color: #ffd700;
                transform: scale(1.05);
            }}
            
            .suspect-id {{
                font-size: 18px;
                font-weight: bold;
                color: #ffd700;
                margin-bottom: 10px;
            }}
            
            .similarity-score {{
                font-size: 16px;
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 20px;
                display: inline-block;
                margin: 5px 0;
            }}
            
            .similarity-high {{ background: linear-gradient(45deg, #28a745, #20903c); }}
            .similarity-medium {{ background: linear-gradient(45deg, #ffc107, #e0a800); color: black; }}
            .similarity-low {{ background: linear-gradient(45deg, #dc3545, #c82333); }}
            
            .timestamp {{
                font-size: 14px;
                color: rgba(255, 255, 255, 0.8);
            }}
            
            .no-images {{
                text-align: center;
                padding: 60px;
                color: rgba(255, 255, 255, 0.7);
                font-size: 18px;
            }}
            
            .refresh-btn {{
                background: linear-gradient(45deg, #28a745, #20903c);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 20px;
            }}
            
            .refresh-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(40, 167, 69, 0.4);
            }}
            
            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.9);
            }}
            
            .modal-content {{
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                margin: 3% auto;
                padding: 20px;
                border-radius: 15px;
                max-width: 80%;
                max-height: 80%;
                overflow: auto;
                border: 2px solid #ffd700;
                text-align: center;
            }}
            
            .modal-image {{
                width: 100%;
                max-width: 600px;
                height: auto;
                border-radius: 10px;
                border: 2px solid #ffd700;
            }}
            
            .close {{
                color: #ffd700;
                float: right;
                font-size: 32px;
                font-weight: bold;
                cursor: pointer;
                line-height: 1;
            }}
            
            .close:hover {{ color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-content">
                <div class="logo">
                    <h1>🖼️ 크롭 이미지 뷰어 - {case_id}</h1>
                </div>
                <div style="background: #dc3545; color: white; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;">RESTRICTED</div>
            </div>
        </div>

        <div class="container">
            <!-- 케이스 정보 -->
            <div class="case-info">
                <h3>📋 수사 케이스: {case_id}</h3>
                <button class="refresh-btn" onclick="loadImages()">🔄 이미지 새로고침</button>
            </div>

            <!-- 이미지 표시 영역 -->
            <div id="imagesContainer">
                <div class="loading">
                    <div class="spinner"></div>
                    <div>크롭 이미지를 로딩 중입니다...</div>
                </div>
            </div>
        </div>

        <!-- 이미지 확대 모달 -->
        <div id="imageModal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeModal()">&times;</span>
                <div>
                    <img id="modalImage" class="modal-image" src="" alt="">
                    <div id="modalInfo" style="margin-top: 15px; color: #ffd700;"></div>
                </div>
            </div>
        </div>

        <script>
            const API_BASE = 'http://localhost:8000';
            const CASE_ID = '{case_id}';

            document.addEventListener('DOMContentLoaded', function() {{
                loadImages();
                
                // 5초마다 자동 새로고침
                setInterval(loadImages, 5000);
            }});

            async function loadImages() {{
                try {{
                    const response = await fetch(`${{API_BASE}}/police/case_report/${{CASE_ID}}`);
                    
                    if (response.ok) {{
                        const result = await response.json();
                        
                        if (result.status === 'incomplete') {{
                            showLoading(`분석 진행률: ${{result.current_progress || 0}}%`);
                        }} else {{
                            displayImages(result);
                        }}
                    }} else if (response.status === 400) {{
                        showLoading('분석이 진행 중입니다...');
                    }} else {{
                        showError('결과를 불러올 수 없습니다');
                    }}
                }} catch (error) {{
                    console.error('이미지 로드 실패:', error);
                    showError('네트워크 오류: ' + error.message);
                }}
            }}

            function displayImages(result) {{
                const container = document.getElementById('imagesContainer');
                const cropImages = result.evidence_package?.cropped_suspect_images || [];
                
                if (cropImages.length === 0) {{
                    container.innerHTML = '<div class="no-images">📷 아직 탐지된 용의자 이미지가 없습니다</div>';
                    return;
                }}
                
                // 정확도 순으로 정렬
                cropImages.sort((a, b) => b.similarity - a.similarity);
                
                let html = `
                    <h3>🎯 탐지된 용의자 이미지 (${{cropImages.length}}개)</h3>
                    <div class="images-grid">
                `;
                
                cropImages.forEach((img, index) => {{
                    const accuracy = (img.similarity * 100).toFixed(1);
                    const accuracyClass = accuracy >= 80 ? 'similarity-high' : accuracy >= 60 ? 'similarity-medium' : 'similarity-low';
                    
                    html += `
                        <div class="image-card">
                            <div class="suspect-id">👤 ${{img.suspect_id}}</div>
                            <img src="data:image/png;base64,${{img.cropped_image}}" 
                                 class="crop-image" 
                                 onclick="openModal('data:image/png;base64,${{img.cropped_image}}', '${{img.suspect_id}}', '${{img.timestamp}}', '${{accuracy}}')"
                                 alt="용의자 ${{img.suspect_id}}">
                            <div>
                                <div class="similarity-score ${{accuracyClass}}">${{accuracy}}% 일치</div>
                                <div class="timestamp">🕐 ${{img.timestamp}}</div>
                            </div>
                        </div>
                    `;
                }});
                
                html += '</div>';
                container.innerHTML = html;
            }}
            
            function openModal(imageSrc, suspectId, timestamp, accuracy) {{
                document.getElementById('modalImage').src = imageSrc;
                document.getElementById('modalInfo').innerHTML = `
                    <h3>🎯 용의자: ${{suspectId}}</h3>
                    <p>📅 탐지 시간: ${{timestamp}}</p>
                    <p>🎯 정확도: ${{accuracy}}%</p>
                `;
                document.getElementById('imageModal').style.display = 'block';
            }}
            
            function closeModal() {{
                document.getElementById('imageModal').style.display = 'none';
            }}
            
            function showLoading(message = '크롭 이미지를 로딩 중입니다...') {{
                document.getElementById('imagesContainer').innerHTML = `
                    <div class="loading">
                        <div class="spinner"></div>
                        <div>${{message}}</div>
                    </div>
                `;
            }}
            
            function showError(message) {{
                document.getElementById('imagesContainer').innerHTML = `
                    <div class="no-images">❌ ${{message}}</div>
                `;
            }}
            
            // 모달 외부 클릭 시 닫기
            window.onclick = function(event) {{
                const modal = document.getElementById('imageModal');
                if (event.target === modal) {{
                    modal.style.display = 'none';
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.get("/police/image_viewer", response_class=HTMLResponse)
async def police_image_viewer_main():
    """메인 이미지 뷰어 (케이스 선택)"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>경찰청 크롭 이미지 뷰어</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%);
                min-height: 100vh;
                color: #fff;
            }
            .header {
                background: rgba(0, 0, 0, 0.3);
                padding: 20px 0;
                border-bottom: 3px solid #ffd700;
                backdrop-filter: blur(10px);
            }
            .header-content {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 20px;
            }
            .logo h1 {
                font-size: 1.8em;
                color: #ffd700;
                font-weight: bold;
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 30px 20px; }
            
            .search-section {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .search-grid {
                display: grid;
                grid-template-columns: 1fr 200px;
                gap: 15px;
                align-items: end;
            }
            
            .form-group { margin-bottom: 15px; }
            .form-label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #ffd700;
            }
            .form-input {
                width: 100%;
                padding: 12px 15px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                font-size: 16px;
            }
            .form-input::placeholder { color: rgba(255, 255, 255, 0.6); }
            .form-input:focus {
                outline: none;
                border-color: #ffd700;
                background: rgba(255, 255, 255, 0.15);
            }
            
            .btn {
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                text-transform: uppercase;
            }
            .btn-primary {
                background: linear-gradient(45deg, #dc3545, #c82333);
                color: white;
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(220, 53, 69, 0.4);
            }
            
            .cases-list {
                display: grid;
                gap: 20px;
                margin-top: 20px;
            }
            
            .case-card {
                background: rgba(255, 255, 255, 0.15);
                border-radius: 15px;
                padding: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .case-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
                border-color: #ffd700;
            }
            
            .case-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            
            .case-id {
                font-size: 18px;
                font-weight: bold;
                color: #ffd700;
            }
            
            .case-info {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }
            
            .info-item {
                background: rgba(0, 0, 0, 0.2);
                padding: 10px;
                border-radius: 8px;
                text-align: center;
            }
            
            .no-cases {
                text-align: center;
                padding: 60px;
                color: rgba(255, 255, 255, 0.7);
                font-size: 18px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-content">
                <div class="logo">
                    <h1>🖼️ 경찰청 크롭 이미지 뷰어</h1>
                </div>
                <div style="background: #dc3545; color: white; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;">RESTRICTED</div>
            </div>
        </div>

        <div class="container">
            <!-- 검색 섹션 -->
            <div class="search-section">
                <h3>🔍 수사 케이스 선택</h3>
                <div class="search-grid">
                    <div class="form-group">
                        <label class="form-label">케이스 ID</label>
                        <input type="text" id="caseIdInput" class="form-input" placeholder="CASE_20250601_143045_a1b2c3d4">
                    </div>
                    <button class="btn btn-primary" onclick="viewCaseImages()">🖼️ 이미지 보기</button>
                </div>
                <div style="margin-top: 15px;">
                    <button class="btn btn-primary" onclick="loadAllCases()" style="background: linear-gradient(45deg, #28a745, #20903c);">📋 전체 케이스 목록</button>
                </div>
            </div>

            <!-- 케이스 목록 -->
            <div id="casesContainer">
                <div class="no-cases">
                    📂 케이스 ID를 입력하거나 전체 케이스 목록을 조회하세요
                </div>
            </div>
        </div>

        <script>
            const API_BASE = 'http://localhost:8000';

            async function viewCaseImages() {
                const caseId = document.getElementById('caseIdInput').value;
                
                if (!caseId) {
                    alert('케이스 ID를 입력하세요');
                    return;
                }
                
                // 이미지 뷰어 페이지로 이동
                window.location.href = `/police/image_viewer/${caseId}`;
            }

            async function loadAllCases() {
                try {
                    const response = await fetch(`${API_BASE}/police/all_cases`);
                    
                    if (response.ok) {
                        const data = await response.json();
                        displayCases(data.cases || []);
                    } else {
                        throw new Error('케이스 목록 조회 실패');
                    }
                } catch (error) {
                    showError('케이스 목록 조회 실패: ' + error.message);
                }
            }

            function displayCases(cases) {
                const container = document.getElementById('casesContainer');
                
                if (cases.length === 0) {
                    container.innerHTML = '<div class="no-cases">📂 등록된 케이스가 없습니다</div>';
                    return;
                }
                
                let html = '<div class="cases-list">';
                
                cases.forEach(caseInfo => {
                    html += `
                        <div class="case-card" onclick="viewCaseImages('${caseInfo.case_id}')">
                            <div class="case-header">
                                <div class="case-id">${caseInfo.case_id}</div>
                                <div>📸 이미지 보기 →</div>
                            </div>
                            <div class="case-info">
                                <div class="info-item">
                                    <strong>위치</strong><br>
                                    ${caseInfo.location || '미지정'}
                                </div>
                                <div class="info-item">
                                    <strong>일시</strong><br>
                                    ${caseInfo.date || '미지정'}
                                </div>
                                <div class="info-item">
                                    <strong>담당 수사관</strong><br>
                                    ${caseInfo.officer_name || '미지정'}
                                </div>
                                <div class="info-item">
                                    <strong>상태</strong><br>
                                    ${getStatusText(caseInfo.status)}
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                html += '</div>';
                container.innerHTML = html;
            }

            function viewCaseImages(caseId) {
                if (caseId) {
                    window.location.href = `/police/image_viewer/${caseId}`;
                } else {
                    viewCaseImages(); // 입력 필드에서 가져오기
                }
            }

            function getStatusText(status) {
                switch(status) {
                    case 'completed': return '✅ 분석 완료';
                    case 'analyzing': return '⏳ 분석 중';
                    case 'failed': return '❌ 실패';
                    default: return '📋 대기 중';
                }
            }

            function showError(message) {
                document.getElementById('casesContainer').innerHTML = `
                    <div class="no-cases">❌ ${message}</div>
                `;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)