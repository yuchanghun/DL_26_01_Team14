import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

from kfashion_dataset import STYLE_CLASSES, CATEGORY_CLASSES, IMG_SIZE

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

if __name__ == '__main__':
    MODEL_DIR = Path(__file__).parent / 'model'
    if len(sys.argv) < 2:
        print('사용법: python predict.py 이미지경로')
        sys.exit(1)
    img_path = sys.argv[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StyleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_DIR / 'model_a_v4.pth', map_location=device, weights_only=True))
    model.eval()

    seg_model = YOLO(str(MODEL_DIR / 'model_seg.pt'))

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert('RGB')

    # 옷 영역 감지
    results = seg_model(img_path, verbose=False)
    boxes = results[0].boxes
    crops = []
    if boxes is not None and len(boxes):
        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            crop = img.crop((x1, y1, x2, y2))
            crops.append(crop)
        print(f'[옷 감지] {len(crops)}개 크롭')
    else:
        crops = [img]
        print('[옷 감지] 감지 실패, 원본 이미지 사용')

    # 각 크롭 예측 후 평균
    all_probs_s = []
    all_probs_c = []
    with torch.no_grad():
        for crop in crops:
            x = transform(crop).unsqueeze(0).to(device)
            out_s, out_c = model(x)
            all_probs_s.append(torch.sigmoid(out_s)[0])
            all_probs_c.append(torch.softmax(out_c, dim=1)[0])

    probs_s = torch.stack(all_probs_s).mean(dim=0)
    probs_c = torch.stack(all_probs_c).mean(dim=0)

    print(f"\n[ {Path(img_path).name} ]")
    print("=== 스타일 (상위 5개) ===")
    for prob, idx in zip(*probs_s.topk(5)):
        print(f"  {STYLE_CLASSES[idx]:16s} {prob.item():.1%}")

    print("\n=== 대분류 ===")
    for prob, idx in zip(*probs_c.topk(3)):
        print(f"  {CATEGORY_CLASSES[idx]:10s} {prob.item():.1%}")
