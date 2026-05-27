import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from kfashion_dataset import STYLE_CLASSES, CATEGORY_CLASSES, IMG_SIZE

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

if __name__ == '__main__':
    MODEL_DIR = Path(__file__).parent / 'model'
    if len(sys.argv) < 2:
        print('사용법: python predict.py 이미지경로')
        sys.exit(1)
    img_path = sys.argv[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StyleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_DIR / 'model_a_cnn.pth', map_location=device))
    model.eval()

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_s, out_c = model(x)
        probs_s = torch.softmax(out_s, dim=1)[0]
        probs_c = torch.softmax(out_c, dim=1)[0]

    print(f"\n[ {Path(img_path).name} ]")
    print("=== 스타일 TOP 5 ===")
    for prob, idx in zip(*probs_s.topk(5)):
        print(f"  {STYLE_CLASSES[idx]:16s} {prob.item():.1%}")

    print("\n=== 대분류 TOP 3 ===")
    for prob, idx in zip(*probs_c.topk(3)):
        print(f"  {CATEGORY_CLASSES[idx]:10s} {prob.item():.1%}")
