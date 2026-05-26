# K-Fashion 딥러닝 기반 개인 맞춤형 패션 추천 시스템

딥러닝기초 14조 팀 프로젝트

## 개요

사용자가 사진을 올리고 키·몸무게·핏 선호도를 입력하면, AI가 스타일을 분석하고 사이즈를 예측해 트렌디한 패션 아이템 TOP 5를 추천하는 시스템입니다.

## 파이프라인

```
사용자 이미지 → [모델 A] 스타일 분류 (24개)
키 / 몸무게 / 핏 선호도 → [모델 B] 사이즈 예측
                        ↓
               스타일 + 사이즈로 후보 필터링
                        ↓
              [모델 C] 트렌드 점수로 정렬 → TOP 5 추천
```

## 모델

### 모델 A — 스타일 분류 (ResNet18)
- K-Fashion 데이터셋 (AI Hub) 학습: 967,806장
- 스타일 24개 + 대분류 5개 멀티태스크 분류
- 검증 정확도: **72.89%** (랜덤 예측 4.17% 대비 17.5배)

### 모델 B — 사이즈 예측 (KNN)
- 의류 통합 데이터셋 (AI Hub) 실측 데이터: 495,561개
- 입력: 키(cm), 몸무게(kg), 핏 선호도 (slim / regular / over)
- 핏 선호도 보정: 인접 사이즈 확률 30% 이상일 때만 이동
- 정확도: **78.22%**

### 모델 C — 트렌드 점수
```
트렌드 점수 = 7일 판매량 × 0.6 + 30일 판매량 × 0.3 + 조회수 × 0.1
```

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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install pandas pillow scikit-learn segmentation-models-pytorch jupyter
```

### 추천 실행
1. `data\` 폴더에 이미지 넣기
2. `패션추천.bat` 더블클릭
3. 파일명 / 키 / 몸무게 / 핏(slim·regular·over) 입력

또는 터미널에서:
```powershell
python run_pipeline.py 내사진.jpg 168 60 regular
```

## 파일 구조

```
├── train.py                  # 모델 A 학습 스크립트
├── train_seg.py              # 세그멘테이션 U-Net 학습 스크립트
├── kfashion_dataset.py       # K-Fashion 데이터 로더
├── run_pipeline.py           # 전체 파이프라인 실행
├── predict.py                # 이미지 단독 스타일 예측
├── 패션추천.bat               # 더블클릭 실행 파일
├── recommend.ps1             # 입력 UI 스크립트
├── 모델A_CNN.ipynb            # 모델 A 학습 노트북
├── 모델C_트렌드.ipynb          # 트렌드 점수 노트북
├── 추천시스템.ipynb            # 통합 파이프라인 노트북
├── 합성데이터생성.ipynb         # 샘플 데이터 생성 노트북
└── data/
    ├── 상품목록.csv            # 샘플 상품 데이터
    ├── 판매데이터.csv          # 샘플 판매 데이터
    ├── model_a_cnn.pth        # 모델 A 가중치
    ├── model_b_knn.pkl        # 모델 B KNN
    └── size_dataset.csv       # 사이즈 학습 데이터
```

## 팀

14조 — 이기열, 유창훈, 심명준
