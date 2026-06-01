import sys, io, requests, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
import plotly.graph_objects as go
import gradio as gr

from kfashion_dataset import STYLE_CLASSES, CATEGORY_CLASSES, IMG_SIZE

MODEL_DIR = Path(__file__).parent / 'model'
DATA_DIR  = Path(__file__).parent / 'data'
SIZE_CLASSES = ['XS', 'S', 'M', 'L', 'XL']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StyleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.backbone      = nn.Sequential(*list(backbone.children())[:-1])
        self.style_head    = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(2048, len(STYLE_CLASSES)))
        self.category_head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(2048, len(CATEGORY_CLASSES)))
    def forward(self, x):
        feat = self.backbone(x)
        return self.style_head(feat), self.category_head(feat)

style_model = StyleCNN().to(device)
style_model.load_state_dict(torch.load(MODEL_DIR / 'model_a_v4.pth', map_location=device, weights_only=True))
style_model.eval()

person_yolo = YOLO(str(MODEL_DIR / 'yolov8n.pt'))

with open(MODEL_DIR / 'model_b_knn.pkl', 'rb') as f:
    knn_data = pickle.load(f)
knn    = knn_data['knn']
scaler = knn_data['scaler']

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

musinsa_orig = pd.read_csv(DATA_DIR / 'musinsa_test_50.csv')
url_map  = dict(zip(musinsa_orig['상품번호'], musinsa_orig['이미지URL']))
link_map = dict(zip(musinsa_orig['상품번호'], musinsa_orig['링크']))

def predict_size(height, weight, fit):
    x     = scaler.transform([[height, weight]])
    proba = knn.predict_proba(x)[0]
    prob_dict = dict(zip(knn.classes_, proba))
    base  = knn.predict(x)[0]
    idx   = SIZE_CLASSES.index(base)
    if fit == 'over' and idx < len(SIZE_CLASSES) - 1:
        nxt = SIZE_CLASSES[idx + 1]
        if prob_dict.get(nxt, 0) >= 0.3:
            return base, nxt
    elif fit == 'slim' and idx > 0:
        nxt = SIZE_CLASSES[idx - 1]
        if prob_dict.get(nxt, 0) >= 0.3:
            return base, nxt
    return base, base

def rank_candidates(candidate_ids, rec_size, top_k=5):
    products = pd.read_csv(DATA_DIR / '상품목록.csv', encoding='utf-8-sig')
    sales    = pd.read_csv(DATA_DIR / '판매데이터.csv', encoding='utf-8-sig')
    df = products.merge(sales, on='product_id')
    def minmax(col): return (col - col.min()) / (col.max() - col.min() + 1e-8)
    df['trend_score'] = 0.6*minmax(df['sales_7d']) + 0.3*minmax(df['sales_30d']) + 0.1*minmax(df['view_count'])
    return (
        df[df['product_id'].isin(candidate_ids) & (df['size'] == rec_size)]
        .drop_duplicates(subset='product_id')
        .sort_values('trend_score', ascending=False)
        .head(top_k)
    )

MALE_STYLES   = {'매니시', '젠더리스', '밀리터리', '스트리트', '힙합', '스포티', '웨스턴', '아방가르드', '펑크'}
FEMALE_STYLES = {'페미닌', '로맨틱', '리조트', '소피스트케이티드', '섹시', '컨트리', '히피', '키치', '프레피'}

def recommend(image, height, weight, fit):
    if image is None:
        return None, "이미지를 업로드하세요.", [], ""

    img = Image.fromarray(image).convert('RGB')
    img_np = np.array(img)

    results = person_yolo(img_np, verbose=False, classes=[0])
    boxes = results[0].boxes
    if boxes and len(boxes):
        x1,y1,x2,y2 = map(int, boxes.xyxy[0].tolist())
        img_crop = img.crop((x1,y1,x2,y2))
    else:
        img_crop = img

    x = transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        out_s, _ = style_model(x)
        probs_s = torch.sigmoid(out_s)[0].cpu().numpy()

    top5_idx   = probs_s.argsort()[::-1][:5]
    top5_names = [STYLE_CLASSES[i] for i in top5_idx]
    top5_vals  = [float(probs_s[i]) for i in top5_idx]

    top2_styles = top5_names[:2]

    # 스타일 바 차트
    colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(5)]
    fig = go.Figure(go.Bar(
        x=top5_vals[::-1], y=top5_names[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f'{v:.1%}' for v in top5_vals[::-1]],
        textposition='outside',
    ))
    fig.update_layout(
        title='스타일 분석 결과',
        xaxis=dict(range=[0, 1.15], tickformat='.0%'),
        height=280, margin=dict(l=10, r=60, t=40, b=10),
        plot_bgcolor='#f8f9fa', paper_bgcolor='white',
    )

    base_size, rec_size = predict_size(height, weight, fit)
    fit_label = {'slim': '슬림핏', 'regular': '레귤러핏', 'over': '오버핏'}[fit]
    size_info = f"예측 사이즈: **{rec_size}** ({fit_label} / {height}cm / {weight}kg)"

    products = pd.read_csv(DATA_DIR / '상품목록.csv', encoding='utf-8-sig')
    mask = products['style'].isin(top2_styles) & (products['size'] == rec_size)
    candidates = products[mask]['product_id'].tolist()
    if not candidates:
        candidates = products[products['style'].isin(top2_styles)]['product_id'].tolist()

    ranked = rank_candidates(candidates, rec_size, top_k=5)

    product_images = []
    product_html = '<div style="display:flex; gap:12px; flex-wrap:wrap;">'
    for _, row in ranked.iterrows():
        pid  = row['product_id']
        name = row['product_name']
        price = f"{int(row['price']):,}원"
        style = row['style']
        link  = link_map.get(pid, '#')
        img_url = url_map.get(pid)
        if img_url:
            try:
                r = requests.get(img_url, timeout=5)
                product_images.append(Image.open(io.BytesIO(r.content)).convert('RGB'))
            except:
                pass
        product_html += f'''
        <div style="width:160px; text-align:center; border:1px solid #eee; border-radius:8px; padding:8px;">
            <img src="{img_url}" style="width:140px; height:140px; object-fit:cover; border-radius:4px;"/>
            <p style="margin:6px 0 2px; font-size:12px; font-weight:bold;">{name[:20]}</p>
            <p style="margin:2px 0; color:#e74c3c; font-weight:bold;">{price}</p>
            <p style="margin:2px 0; color:#888; font-size:11px;">{style} · {row["size"]}</p>
            <a href="{link}" target="_blank" style="font-size:11px; color:#3498db;">무신사 바로가기</a>
        </div>'''
    product_html += '</div>'

    return fig, size_info, product_images, product_html

with gr.Blocks(title='K-Fashion 추천', theme=gr.themes.Soft()) as demo:
    gr.Markdown('# 👗 K-Fashion 스타일 추천 시스템\n삼육대학교 인공지능융합학과 14조')
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(label='패션 사진 업로드', height=300, sources=['upload', 'clipboard'])
            with gr.Row():
                height_in = gr.Slider(140, 200, value=165, step=1, label='키 (cm)')
                weight_in = gr.Slider(40, 120, value=60,  step=1, label='몸무게 (kg)')
            fit_in = gr.Radio(['slim', 'regular', 'over'], value='regular',
                              label='핏 선택', info='slim=슬림핏 / regular=레귤러핏 / over=오버핏')
            btn = gr.Button('추천 받기 ▶', variant='primary', size='lg')
        with gr.Column(scale=1):
            chart_out = gr.Plot(label='스타일 분석')
            size_out  = gr.Markdown()

    gr.Markdown('### 추천 상품 TOP 5')
    with gr.Row():
        gallery  = gr.Gallery(label='', columns=5, height=220, show_label=False)
    html_out = gr.HTML()

    btn.click(fn=recommend,
              inputs=[img_input, height_in, weight_in, fit_in],
              outputs=[chart_out, size_out, gallery, html_out])

if __name__ == '__main__':
    demo.launch(share=True)
