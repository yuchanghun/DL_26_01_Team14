# K-Fashion 딥러닝 기반 개인 맞춤형 패션 추천 시스템

삼육대학교 인공지능융합학과 14조 (이기열, 유창훈, 심명준)

## 개요

사용자가 사진을 업로드하고 키·몸무게·핏 선호도를 입력하면, AI가 스타일을 분석하고 사이즈를 예측해 무신사 트렌드 아이템 TOP 5를 추천하는 시스템입니다.

## 파이프라인

```
사용자 이미지 → [YOLOv8n] 사람 감지 및 크롭
                         ↓
               [모델 A] 스타일 분류 (23개) + 대분류 (4개)
키 / 몸무게 / 핏 → [모델 B] 사이즈 예측 (KNN)
                         ↓
                스타일 + 사이즈로 후보 필터링
                         ↓
               [트렌드 점수] 정렬 → TOP 5 추천
```

## 모델

### 모델 A — 스타일 분류 (ResNet50 + BCEWithLogitsLoss)
- K-Fashion 데이터셋 (AI Hub): 963,406장 학습 / 120,429장 검증
- 스타일 23개 멀티라벨 + 대분류 4개 멀티태스크
- pos_weight로 클래스 불균형 처리 (clamp max=20)
- 서브스타일 재배정: 스트리트 56.8% → 42.3%로 불균형 개선
- LR 리셋 (Cosine Annealing Warm Restart) 적용

| 버전 | 설명 | Val R@1 | Val R@3 |
|------|------|---------|---------|
| v2 | 기본 학습 (15 epoch) | 53.42% | 89.76% |
| v3 | 서브스타일 재배정 적용 | 57.05% | 90.31% |
| v4 | v3 가중치 + LR 리셋 (15 epoch 추가) | **58.70%** | **91.35%** |

### 모델 B — 사이즈 예측 (KNN)
- 입력: 키(cm), 몸무게(kg), 핏 선호도 (slim / regular / over)
- 핏 보정: 인접 사이즈 확률 30% 이상일 때만 이동

### 트렌드 점수
```
트렌드 점수 = 7일 판매량 × 0.6 + 30일 판매량 × 0.3 + 조회수 × 0.1
```

## 설치 및 실행

### 환경 설정
```bash
conda create -n kfashion python=3.11
conda activate kfashion
pip install torch torchvision ultralytics gradio plotly pandas scikit-learn pillow requests
```

### 모델 파일
`model/` 폴더에 아래 파일 필요 (별도 전달):
- `model_a_v4.pth` — 스타일 분류 모델 (90MB)
- `model_b_knn.pkl` — 사이즈 예측 모델
- `yolov8n.pt` — 사람 감지
- `model_seg.pt` — 옷 영역 감지

### 웹 앱 실행
```bash
conda activate kfashion
python app.py
```
브라우저에서 http://localhost:7860 접속

### CLI 실행
```bash
python run_pipeline.py 이미지파일명.jpg 키(cm) 몸무게(kg) [slim/regular/over]
# 예시
python run_pipeline.py 사진.jpg 168 60 over
```

## 데이터

| 데이터 | 출처 | 용도 |
|--------|------|------|
| K-Fashion 이미지 (963,406장) | AI Hub | 모델 A 학습 |
| 무신사 인기 상품 50개 | 무신사 크롤링 | 추천 상품 DB |

## 한계 및 고찰

- AI Hub K-Fashion 데이터셋 라벨 품질 문제 (스트리트·펑크 라벨 노이즈 확인)
- 학습 분포(옷 크롭)와 실제 추론 분포(전신 이미지) 간 도메인 갭
- 무신사 상품이 상의 위주로 편향
