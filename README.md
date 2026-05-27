# K-Fashion 딥러닝 기반 개인 맞춤형 패션 추천 시스템

딥러닝기초 14조 팀 프로젝트

## 개요

사용자가 사진을 올리고 키·몸무게·핏 선호도를 입력하면, AI가 스타일을 분석하고 사이즈를 예측해 트렌디한 패션 아이템 TOP 5를 추천하는 시스템입니다.

## 파이프라인

```
사용자 이미지 → [YOLO] 사람 감지 및 크롭
                        ↓
              [모델 A] 스타일 분류 (23개) + 대분류 (4개)
키 / 몸무게 / 핏 선호도 → [모델 B] 사이즈 예측
                        ↓
               스타일 + 사이즈로 후보 필터링
                        ↓
              [모델 C] 트렌드 점수로 정렬 → TOP 5 추천
```

## 모델

### 모델 A — 스타일 분류 (ResNet18)
- K-Fashion 데이터셋 (AI Hub) 학습: 967,806장
- 스타일 23개 + 대분류 4개 멀티태스크 분류
- 검증 정확도: **72.89%** (랜덤 예측 4.35% 대비 16.7배)

### 모델 B — 사이즈 예측 (KNN)
- 의류 통합 데이터셋 (AI Hub) 실측 데이터: 495,561개
- 입력: 키(cm), 몸무게(kg), 핏 선호도 (slim / regular / over)
- 핏 선호도 보정: 인접 사이즈 확률 30% 이상일 때만 이동
- 정확도: **78.22%**

### 모델 C — 트렌드 점수
```
트렌드 점수 = 7일 판매량 × 0.6 + 30일 판매량 × 0.3 + 조회수 × 0.1
```

### YOLO 세그멘테이션 — 옷 감지
- YOLOv8n-seg 기반, K-Fashion 데이터로 파인튜닝
- 이미지에서 옷 영역만 크롭해 모델 A 정확도 향상

## 사용 데이터

| 데이터 | 출처 | 용도 |
|--------|------|------|
| K-Fashion 이미지 (967,806장) | AI Hub | 모델 A 학습 |
| 의류 통합 데이터 (397,345장) | AI Hub | 모델 B 실측 데이터, 세그멘테이션 |

## 실행 방법

### 환경 설정
```powershell
conda create -n kfashion python=3.11
conda activate kfashion

# GPU (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip install torch torchvision

pip install pandas pillow scikit-learn ultralytics jupyter
```

### 추천 실행
1. `data\` 폴더에 이미지 넣기
2. `패션추천.bat` 더블클릭
3. 파일명 / 키 / 몸무게 / 핏(slim·regular·over) 입력

또는 터미널에서:
```powershell
conda activate kfashion
python run_pipeline.py 내사진.jpg 168 60 regular
```

## 파일 구조

```
├── train.py                  # 모델 A 학습 스크립트
├── train_seg.py              # 세그멘테이션 YOLOv8-seg 학습 스크립트
├── kfashion_dataset.py       # K-Fashion 데이터 로더
├── run_pipeline.py           # 전체 파이프라인 실행
├── predict.py                # 이미지 단독 스타일 예측
├── eval_compare.py           # 모델 평가 및 비교
├── prepare_balanced.py       # 클래스 균형 데이터 준비
├── plot_training.py          # 학습 곡선 시각화
├── 패션추천.bat               # 더블클릭 실행 파일
├── recommend.ps1             # 입력 UI 스크립트
├── model/
│   ├── model_a_cnn.pth       # 모델 A 가중치
│   ├── model_a_balanced.pth  # 모델 A 균형 학습 가중치
│   ├── model_b_knn.pkl       # 모델 B KNN
│   ├── model_seg.pt          # YOLO 세그멘테이션 (옷 감지)
│   └── yolov8n.pt            # YOLO (사람 감지)
├── result/
│   ├── train_log.txt         # 학습 로그
│   └── training_comparison.png  # 학습 곡선 그래프
└── data/
    ├── 상품목록.csv            # 샘플 상품 데이터
    ├── 판매데이터.csv          # 샘플 판매 데이터
    └── size_dataset.csv       # 사이즈 학습 데이터
```

## 팀

14조 — 이기열, 유창훈, 심명준
