import sys, requests, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import pandas as pd
import numpy as np
import io
from PIL import Image
from ultralytics import YOLO
import plotly.graph_objects as go
import gradio as gr

from kfashion_dataset import STYLE_CLASSES, CATEGORY_CLASSES, IMG_SIZE

MODEL_DIR = Path(__file__).parent / 'model'
DATA_DIR  = Path(__file__).parent / 'data'
SIZE_CLASSES = ['XS', 'S', 'M', 'L', 'XL', '2XL']

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
    _knn   = pickle.load(f)
knn    = _knn['knn']
scaler = _knn['scaler']

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

products_df = pd.read_csv(DATA_DIR / 'musinsa_products.csv', encoding='utf-8-sig')
sales_df    = pd.read_csv(DATA_DIR / 'musinsa_sales.csv',    encoding='utf-8-sig')

def predict_size(height, weight, fit):
    x     = scaler.transform([[height, weight]])
    proba = knn.predict_proba(x)[0]
    prob_dict = dict(zip(knn.classes_, proba))
    base  = knn.predict(x)[0]
    idx   = SIZE_CLASSES.index(base) if base in SIZE_CLASSES else 3
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
    df = products_df.merge(sales_df, on='product_id')
    def minmax(col): return (col - col.min()) / (col.max() - col.min() + 1e-8)
    df['trend_score'] = 0.3*minmax(df['sales']) + 0.7*minmax(df['view_count'])
    return (
        df[df['product_id'].isin(candidate_ids) & (df['size'] == rec_size)]
        .drop_duplicates(subset='product_id')
        .sort_values('trend_score', ascending=False)
        .head(top_k)
    )

CAT_MAP = {
    '전체':        None,
    '상의':        ['상의'],
    '하의':        ['하의'],
    '아우터':       ['아우터'],
    '원피스/스커트':  ['원피스/스커트', '원피스', '스커트'],
}

def analyze_style(image):
    if image is None:
        return None, gr.CheckboxGroup(choices=[], value=[])

    img = Image.fromarray(image).convert('RGB')
    img_np = np.array(img)

    results = person_yolo(img_np, verbose=False, classes=[0])
    boxes = results[0].boxes
    img_crop = img.crop(tuple(map(int, boxes.xyxy[0].tolist()))) if (boxes and len(boxes)) else img

    x = transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        out_s, _ = style_model(x)
        probs_s = torch.sigmoid(out_s)[0].cpu().numpy()

    top5_idx   = probs_s.argsort()[::-1][:5]
    top5_names = [STYLE_CLASSES[i] for i in top5_idx]
    top5_vals  = [float(probs_s[i]) for i in top5_idx]

    bar_colors = ['#0f3460' if i == 0 else '#3498db' if i == 1 else '#aac4de' for i in range(5)]
    fig = go.Figure(go.Bar(
        x=top5_vals[::-1], y=top5_names[::-1],
        orientation='h',
        marker=dict(color=bar_colors[::-1], line=dict(width=0)),
        text=[f'{v:.1%}' for v in top5_vals[::-1]], textposition='outside',
        textfont=dict(size=13, color='#333'),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1.2], tickformat='.0%', gridcolor='#f0f2f5', zeroline=False),
        yaxis=dict(tickfont=dict(size=13)),
        height=260, margin=dict(l=10, r=70, t=16, b=10),
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        font=dict(family='Segoe UI, Apple SD Gothic Neo, sans-serif'),
    )

    return fig, gr.CheckboxGroup(choices=top5_names, value=top5_names[:2])


def get_recommendations(selected_styles, height, weight, fit, category, gender):
    if not selected_styles:
        return "스타일을 먼저 선택하세요.", "", ""

    _, rec_size = predict_size(height, weight, fit)
    fit_label = {'slim': '슬림핏', 'regular': '레귤러핏', 'over': '오버핏'}[fit]
    size_info = f"예측 사이즈: **{rec_size}** ({fit_label} / {height}cm / {weight}kg)"

    prod = products_df.copy()
    cats = CAT_MAP.get(category)
    if cats:
        prod = prod[prod['category'].isin(cats)]
    if gender != '전체':
        prod = prod[prod['gender'].isin([gender, '공용'])]

    mask = prod['style1'].isin(selected_styles) & (prod['size'] == rec_size)
    candidates = prod[mask]['product_id'].tolist()
    if not candidates:
        candidates = prod[prod['style1'].isin(selected_styles)]['product_id'].tolist()
    if not candidates:
        candidates = prod[prod['size'] == rec_size]['product_id'].tolist()

    top5    = rank_candidates(candidates, rec_size, top_k=10)
    top15   = rank_candidates(candidates, rec_size, top_k=20)

    product_html = '<div style="display:flex;gap:14px;flex-wrap:nowrap;overflow-x:auto;padding:4px 0 12px;">'
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        img_url = row.get('image_url', '')
        link    = row.get('link', '#')
        name    = str(row['product_name'])
        price   = f"{int(row['price']):,}원"
        style   = row.get('style1', '')
        cat     = row.get('category', '')
        badge_bg = '#0f3460' if rank == 1 else '#3498db' if rank == 2 else '#6c757d'
        product_html += f'''
        <div style="width:190px;border-radius:12px;overflow:hidden;background:#fff;border:1px solid #e8eaed;box-shadow:0 2px 10px rgba(0,0,0,0.06);font-family:\'Segoe UI\',sans-serif;">
            <div style="position:relative;">
                <img src="{img_url}" style="width:100%;height:190px;object-fit:cover;display:block;"/>
                <span style="position:absolute;top:8px;left:8px;background:{badge_bg};color:#fff;font-size:11px;font-weight:700;padding:3px 8px;border-radius:20px;">#{rank}</span>
            </div>
            <div style="padding:12px;">
                <p style="margin:0 0 4px;font-size:12px;font-weight:700;color:#1a1a2e;line-height:1.4;">{name[:28]}</p>
                <p style="margin:0 0 6px;font-size:15px;font-weight:800;color:#0f3460;">{price}</p>
                <p style="margin:0 0 10px;font-size:10px;color:#888;">{style} · {cat} · {row["size"]}</p>
                <a href="{link}" target="_blank" style="display:block;text-align:center;padding:7px 0;background:#0f3460;color:#fff;border-radius:8px;font-size:12px;font-weight:600;text-decoration:none;">무신사 바로가기</a>
            </div>
        </div>'''
    product_html += '</div>'

    rank_html = '''<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-size:13px;font-family:\'Segoe UI\',sans-serif;">
<thead><tr style="background:#f8f9fa;border-bottom:2px solid #dee2e6;">
<th style="padding:10px 12px;text-align:center;color:#495057;font-weight:600;width:40px;">#</th>
<th style="padding:10px 12px;text-align:left;color:#495057;font-weight:600;">상품명</th>
<th style="padding:10px 12px;text-align:left;color:#495057;font-weight:600;white-space:nowrap;">브랜드</th>
<th style="padding:10px 12px;text-align:left;color:#495057;font-weight:600;white-space:nowrap;">카테고리</th>
<th style="padding:10px 12px;text-align:left;color:#495057;font-weight:600;white-space:nowrap;">스타일</th>
<th style="padding:10px 12px;text-align:center;color:#495057;font-weight:600;white-space:nowrap;">사이즈</th>
<th style="padding:10px 12px;text-align:right;color:#495057;font-weight:600;white-space:nowrap;">가격</th>
<th style="padding:10px 12px;text-align:center;color:#495057;font-weight:600;white-space:nowrap;">링크</th>
</tr></thead><tbody>'''
    for i, (_, row) in enumerate(top15.iterrows(), 1):
        bg = '#f0f7ff' if i % 2 == 0 else '#ffffff'
        rank_badge = f'<span style="display:inline-block;width:24px;height:24px;line-height:24px;text-align:center;border-radius:50%;background:{"#0f3460" if i==1 else "#3498db" if i==2 else "#6c757d" if i==3 else "#dee2e6"};color:{"#fff" if i<=3 else "#333"};font-size:11px;font-weight:700;">{i}</span>'
        rank_html += f'''<tr style="background:{bg};border-bottom:1px solid #f0f2f5;" onmouseover="this.style.background=\'#e8f4fd\'" onmouseout="this.style.background=\'{bg}\'">
<td style="padding:9px 12px;text-align:center;">{rank_badge}</td>
<td style="padding:9px 12px;color:#1a1a2e;font-weight:500;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{row['product_name']}">{str(row['product_name'])[:35]}</td>
<td style="padding:9px 12px;color:#555;white-space:nowrap;">{row.get('brand','')}</td>
<td style="padding:9px 12px;color:#555;white-space:nowrap;">{row.get('category','')}</td>
<td style="padding:9px 12px;color:#555;white-space:nowrap;">{row.get('style1','')}</td>
<td style="padding:9px 12px;text-align:center;color:#555;font-weight:600;">{row['size']}</td>
<td style="padding:9px 12px;text-align:right;color:#0f3460;font-weight:700;white-space:nowrap;">{int(row['price']):,}원</td>
<td style="padding:9px 12px;text-align:center;"><a href="{row.get('link','#')}" target="_blank" style="display:inline-block;padding:4px 10px;background:#0f3460;color:#fff;border-radius:6px;font-size:11px;font-weight:600;text-decoration:none;white-space:nowrap;">바로가기</a></td>
</tr>'''
    rank_html += '</tbody></table></div>'

    return size_info, product_html, rank_html

CSS = """
/* ── 전체 배경 ── */
body, .gradio-container { background: #f0f2f5 !important; font-family: 'Segoe UI', 'Apple SD Gothic Neo', sans-serif; }

/* ── 헤더 ── */
#header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 20px;
    color: white;
}
#header h1 { font-size: 2rem; font-weight: 800; margin: 0 0 4px; color: #fff; }
#header p  { font-size: 0.9rem; color: #8892b0; margin: 0; }

/* ── 카드 공통 ── */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1px solid #e8eaed;
}
.card-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6c757d;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 2px solid #f0f2f5;
}

/* ── 입력 패널 ── */
#input-panel { background: #ffffff; border-radius: 14px; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.07); border: 1px solid #e8eaed; }

/* ── 버튼 ── */
#rec-btn { background: #0f3460 !important; border: none !important; border-radius: 10px !important; font-size: 1rem !important; font-weight: 700 !important; height: 52px !important; letter-spacing: 0.04em; }
#rec-btn:hover { background: #16213e !important; }
#analyze-btn { border-radius: 8px !important; font-size: 0.85rem !important; font-weight: 600 !important; margin-bottom: 4px; }

/* ── 사이즈 결과 ── */
#size-result { background: #eef6ff; border-radius: 10px; padding: 14px 18px; border-left: 4px solid #3498db; }

/* ── 랭킹 테이블 ── */
#rank-table table { border-collapse: collapse; width: 100%; font-size: 13px; }
#rank-table thead th { background: #f8f9fa; color: #495057; font-weight: 600; padding: 10px 12px; border-bottom: 2px solid #dee2e6; }
#rank-table tbody tr:hover { background: #f0f7ff; }
#rank-table tbody td { padding: 9px 12px; border-bottom: 1px solid #f0f2f5; color: #333; }

/* ── 구분선 제거 ── */
.divider { border: none; border-top: 2px solid #e8eaed; margin: 20px 0; }
"""

with gr.Blocks(title='K-Fashion 추천') as demo:

    # ── 헤더
    gr.HTML('''
    <div id="header">
        <h1>K-Fashion 스타일 추천</h1>
        <p>삼육대학교 인공지능융합학과 14조 &nbsp;·&nbsp; ResNet50 + KNN + YOLOv8</p>
    </div>
    ''')

    # ── 상단: 입력 / 분석 결과
    with gr.Row(equal_height=False):
        # 왼쪽: 입력 패널
        with gr.Column(scale=1, min_width=320, elem_id='input-panel'):
            gr.HTML('<div class="card-title">입력 정보</div>')
            img_input = gr.Image(label='패션 사진 업로드', height=240, sources=['upload', 'clipboard'])
            analyze_btn = gr.Button('스타일 분석', variant='secondary', size='sm', elem_id='analyze-btn')
            with gr.Row():
                height_in = gr.Slider(140, 200, value=165, step=1, label='키 (cm)')
                weight_in = gr.Slider(40, 120, value=60,  step=1, label='몸무게 (kg)')
            with gr.Row():
                fit_in    = gr.Radio(['slim', 'regular', 'over'], value='regular', label='핏 선택')
                gender_in = gr.Radio(['전체', '남성', '여성'],      value='전체',    label='성별')
            cat_in = gr.Radio(['전체', '상의', '하의', '아우터', '원피스/스커트'], value='전체', label='카테고리')
            btn = gr.Button('추천 받기', variant='primary', size='lg', elem_id='rec-btn')

        # 오른쪽: 차트 + 스타일 선택 + 사이즈
        with gr.Column(scale=1, min_width=380):
            with gr.Group(elem_classes='card'):
                gr.HTML('<div class="card-title">스타일 분석 결과</div>')
                chart_out = gr.Plot(label='', show_label=False)
                gr.HTML('<div style="border-top:1px solid #f0f2f5;margin:10px 0 12px;"></div>')
                gr.HTML('<div class="card-title" style="margin-bottom:8px;">추천에 사용할 스타일 선택</div>')
                style_selector = gr.CheckboxGroup(choices=[], value=[], label='', show_label=False)
                gr.HTML('<div style="border-top:1px solid #f0f2f5;margin:10px 0 12px;"></div>')
                gr.HTML('<div class="card-title" style="margin-bottom:8px;">사이즈 추천</div>')
                size_out = gr.Markdown(elem_id='size-result')

    # ── 하단: 추천 상품 (전체 너비)
    gr.HTML('<hr class="divider"/>')
    with gr.Group(elem_classes='card'):
        gr.HTML('<div class="card-title">추천 TOP 10</div>')
        html_out = gr.HTML()

    # ── 최하단: 랭킹 테이블
    with gr.Group(elem_classes='card'):
        gr.HTML('<div class="card-title">트렌드 랭킹 TOP 20</div>')
        rank_out = gr.HTML()

    analyze_btn.click(fn=analyze_style,
                      inputs=[img_input],
                      outputs=[chart_out, style_selector])

    btn.click(fn=get_recommendations,
              inputs=[style_selector, height_in, weight_in, fit_in, cat_in, gender_in],
              outputs=[size_out, html_out, rank_out])

if __name__ == '__main__':
    demo.launch(share=True, css=CSS)
