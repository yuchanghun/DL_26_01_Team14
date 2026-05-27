"""K-Fashion 패션 추천 파이프라인"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from kfashion_dataset import STYLE_CLASSES, CATEGORY_CLASSES, IMG_SIZE

DATA_DIR  = Path(__file__).parent / 'data'
MODEL_DIR = Path(__file__).parent / 'model'

SIZE_CLASSES = ['XS', 'S', 'M', 'L', 'XL']
FIT_OFFSET   = {'slim': -1, 'regular': 0, 'over': 1}

EVAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class StyleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.style_head    = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, len(STYLE_CLASSES)))
        self.category_head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, len(CATEGORY_CLASSES)))
    def forward(self, x):
        feat = self.backbone(x)
        return self.style_head(feat), self.category_head(feat)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_a = StyleCNN()
model_a.load_state_dict(torch.load(MODEL_DIR / 'model_a_cnn.pth', map_location=device, weights_only=True))
model_a.to(device).eval()

with open(MODEL_DIR / 'model_b_knn.pkl', 'rb') as f:
    knn_data = pickle.load(f)
knn     = knn_data['knn']
scaler  = knn_data['scaler']

model_person = YOLO(str(MODEL_DIR / 'yolov8n.pt'))

print(f'디바이스: {device}')
print('모델 A (ResNet18) 로드 완료')
print('모델 B (KNN)      로드 완료')
print('사람 감지 (YOLOv8n) 로드 완료')

def predict_size(height_cm, weight_kg, fit='regular'):
    x      = scaler.transform([[height_cm, weight_kg]])
    proba  = knn.predict_proba(x)[0]
    prob_dict = dict(zip(knn.classes_, proba))
    base   = knn.predict(x)[0]
    idx    = SIZE_CLASSES.index(base)

    if fit == 'over' and idx < len(SIZE_CLASSES) - 1:
        next_size = SIZE_CLASSES[idx + 1]
        if prob_dict.get(next_size, 0) >= 0.3:
            return base, next_size
    elif fit == 'slim' and idx > 0:
        next_size = SIZE_CLASSES[idx - 1]
        if prob_dict.get(next_size, 0) >= 0.3:
            return base, next_size

    return base, base

def rank_candidates(candidate_ids, top_k=5):
    products = pd.read_csv('data/상품목록.csv', encoding='utf-8-sig')
    sales    = pd.read_csv('data/판매데이터.csv', encoding='utf-8-sig')
    df = products.merge(sales, on='product_id')
    def minmax(col): return (col - col.min()) / (col.max() - col.min() + 1e-8)
    df['trend_score'] = 0.6*minmax(df['sales_7d']) + 0.3*minmax(df['sales_30d']) + 0.1*minmax(df['view_count'])
    return (
        df[df['product_id'].isin(candidate_ids)]
        .sort_values('trend_score', ascending=False)
        .head(top_k)
    )

def recommend(height_cm, weight_kg, fit='regular', img_path=None, top_k=5):
    # Step 1. 모델 A — 스타일 분류
    if img_path:
        img = Image.open(img_path).convert('RGB')
        results = model_person(img_path, verbose=False, classes=[0])
        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            x1, y1, x2, y2 = map(int, boxes.xyxy[0].tolist())
            img = img.crop((x1, y1, x2, y2))
            print('[사람 감지] 전신 크롭 완료')
        else:
            print('[사람 감지] 감지 실패, 원본 이미지 사용')
        image_tensor = EVAL_TRANSFORM(img).unsqueeze(0)
    else:
        image_tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        out_s, out_c = model_a(image_tensor.to(device))
        style_probs = torch.softmax(out_s, dim=1).squeeze().cpu().numpy()
        cat_probs   = torch.softmax(out_c, dim=1).squeeze().cpu().numpy()

    top_styles = [STYLE_CLASSES[i] for i in style_probs.argsort()[::-1][:2]]
    top_cat    = CATEGORY_CLASSES[cat_probs.argmax()]

    print('\n[모델 A] 스타일 분석 상위 5개:')
    for i in style_probs.argsort()[::-1][:5]:
        bar = '#' * int(style_probs[i] * 20)
        print(f'  {STYLE_CLASSES[i]:16s} {bar:<20s} {style_probs[i]:.1%}')
    print(f'  대분류 예측: {top_cat} ({cat_probs.max():.1%})')

    # Step 2. 모델 B — 사이즈 예측 + 핏 보정
    base_size, rec_size = predict_size(height_cm, weight_kg, fit)
    fit_label = {'slim': '슬림핏', 'regular': '레귤러핏', 'over': '오버핏'}[fit]
    if base_size == rec_size:
        print(f'\n[모델 B] 사이즈 예측: {rec_size}  ({fit_label} / 키 {height_cm}cm / {weight_kg}kg)')
    else:
        print(f'\n[모델 B] 사이즈 예측: {base_size} → {rec_size}  ({fit_label} 보정 / 키 {height_cm}cm / {weight_kg}kg)')

    # Step 3. 후보 필터링
    products = pd.read_csv('data/상품목록.csv', encoding='utf-8-sig')
    mask     = products['style'].isin(top_styles) & (products['size'] == rec_size)
    candidates = products[mask]['product_id'].tolist()
    if not candidates:
        candidates = products[products['style'].isin(top_styles)]['product_id'].tolist()
    print(f'\n[필터링] 스타일: {top_styles} / 사이즈: {rec_size} → 후보 {len(candidates)}개')

    # Step 4. 모델 C — 트렌드 정렬
    ranked = rank_candidates(candidates, top_k)

    print(f'\n{"="*60}')
    print(f'  최종 추천 TOP {top_k}  (트렌드 점수 높은 순)')
    print(f'{"="*60}')
    print(ranked[['product_name', 'style', 'size', 'price', 'trend_score']].to_string(index=False))
    return ranked

# ── 실행: python run_pipeline.py 파일명 키 몸무게 핏
# 핏: slim / regular / over  (생략 시 regular)
# 예) python run_pipeline.py 내사진.jpg 168 60 over
IMG_FOLDER = DATA_DIR

if len(sys.argv) >= 4:
    img  = str(IMG_FOLDER / sys.argv[1])
    h    = float(sys.argv[2])
    w    = float(sys.argv[3])
    fit  = sys.argv[4] if len(sys.argv) == 5 else 'regular'
    recommend(height_cm=h, weight_kg=w, fit=fit, img_path=img)
else:
    print('\n사용법: python run_pipeline.py 파일명 키(cm) 몸무게(kg) [slim/regular/over]')
    print('예시:   python run_pipeline.py 내사진.jpg 168 60 over')
    print(f'이미지 폴더: {IMG_FOLDER}')
