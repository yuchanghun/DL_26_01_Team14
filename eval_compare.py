"""모델 비교 및 전처리 효과 평가
비교 항목:
  - 기존 모델 (model_a_cnn.pth)    vs 균형 모델 (model_a_balanced.pth)
  - 전처리 없음 / 사람 크롭 / 옷 크롭(첫번째) / 옷 크롭(평균)
"""
import sys, random
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from kfashion_dataset import STYLE_CLASSES, CATEGORY_CLASSES, IMG_SIZE

DATA_DIR         = Path(__file__).parent / 'data'
MODEL_DIR        = Path(__file__).parent / 'model'
VAL_ROOT         = Path(r'C:\딥러닝데이터\KFashion_balanced\Validation')
SAMPLE_PER_CLASS = 50
RANDOM_SEED      = 42

EVAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class StyleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.backbone      = nn.Sequential(*list(backbone.children())[:-1])
        self.style_head    = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, len(STYLE_CLASSES)))
        self.category_head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, len(CATEGORY_CLASSES)))

    def forward(self, x):
        feat = self.backbone(x)
        return self.style_head(feat), self.category_head(feat)


def load_model(path, device):
    m = StyleCNN()
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return m.to(device).eval()


def sample_images():
    random.seed(RANDOM_SEED)
    img_dir = VAL_ROOT / '원천데이터'
    samples = []
    for style in STYLE_CLASSES:
        style_dir = img_dir / style
        if not style_dir.exists():
            continue
        imgs = list(style_dir.glob('*.jpg'))
        chosen = random.sample(imgs, min(SAMPLE_PER_CLASS, len(imgs)))
        idx = STYLE_CLASSES.index(style)
        samples.extend([(p, idx, style) for p in chosen])
    return samples


def get_person_crop(img, img_path, yolo_person):
    results = yolo_person(str(img_path), verbose=False, classes=[0])
    boxes = results[0].boxes
    if boxes is not None and len(boxes):
        x1, y1, x2, y2 = map(int, boxes.xyxy[0].tolist())
        return img.crop((x1, y1, x2, y2))
    return img


def get_clothing_crops(img, img_path, yolo_seg):
    """감지된 옷 전부 크롭해서 리스트로 반환, 없으면 원본"""
    results = yolo_seg(str(img_path), verbose=False)
    boxes = results[0].boxes
    if boxes is not None and len(boxes):
        crops = []
        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            crops.append(img.crop((x1, y1, x2, y2)))
        return crops
    return [img]


def infer_single(model, img, device):
    """단일 이미지 → 스타일 softmax 확률 반환"""
    tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out_s, _ = model(tensor)
        return torch.softmax(out_s, dim=1).squeeze()


def eval_accuracy(model, samples, device, mode='none', yolo_person=None, yolo_seg=None):
    """
    mode: 'none' | 'person' | 'clothing_first' | 'clothing_avg'
    """
    correct = 0
    per_class = {s: [0, 0] for s in STYLE_CLASSES}

    for img_path, label, style in samples:
        img = Image.open(img_path).convert('RGB')

        if mode == 'none':
            pred = infer_single(model, img, device).argmax().item()

        elif mode == 'person':
            cropped = get_person_crop(img, img_path, yolo_person)
            pred = infer_single(model, cropped, device).argmax().item()

        elif mode == 'clothing_first':
            crops = get_clothing_crops(img, img_path, yolo_seg)
            pred = infer_single(model, crops[0], device).argmax().item()

        elif mode == 'clothing_avg':
            crops = get_clothing_crops(img, img_path, yolo_seg)
            probs = torch.stack([infer_single(model, c, device) for c in crops])
            pred = probs.mean(dim=0).argmax().item()

        per_class[style][1] += 1
        if pred == label:
            correct += 1
            per_class[style][0] += 1

    return correct / len(samples), per_class


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'디바이스: {device}')

    samples = sample_images()
    print(f'샘플: {len(samples)}장 (클래스당 최대 {SAMPLE_PER_CLASS}장)\n')

    yolo_person = YOLO(str(MODEL_DIR / 'yolov8n.pt'))
    seg_path = MODEL_DIR / 'model_seg.pt'
    yolo_seg = YOLO(str(seg_path)) if seg_path.exists() else None

    modes = [('전처리없음', 'none')]
    modes.append(('사람크롭', 'person'))
    if yolo_seg:
        modes.append(('옷크롭(첫번째)', 'clothing_first'))
        modes.append(('옷크롭(평균)', 'clothing_avg'))
    else:
        print('model_seg.pt 없음 — 옷크롭 스킵\n')

    model_configs = []
    if (MODEL_DIR / 'model_a_cnn.pth').exists():
        model_configs.append(('기존모델(불균형)', MODEL_DIR / 'model_a_cnn.pth'))
    if (MODEL_DIR / 'model_a_balanced.pth').exists():
        model_configs.append(('균형모델', MODEL_DIR / 'model_a_balanced.pth'))

    if not model_configs:
        print('평가할 모델이 없습니다.')
        raise SystemExit

    # 결과 수집
    all_results = {}
    for model_name, path in model_configs:
        print(f'\n[{model_name}] 평가 중...')
        m = load_model(path, device)
        all_results[model_name] = {}
        for mode_name, mode in modes:
            acc, pc = eval_accuracy(m, samples, device, mode=mode,
                                    yolo_person=yolo_person, yolo_seg=yolo_seg)
            all_results[model_name][mode_name] = (acc, pc)
            print(f'  {mode_name:<14}: {acc:.2%}')

    # 요약 테이블
    col_w = 14
    print(f'\n{"="*70}')
    print(f'  {"모델":<18}' + ''.join(f'{n:>{col_w}}' for n, _ in modes))
    print(f'{"="*70}')
    for model_name, _ in model_configs:
        row = f'  {model_name:<18}'
        for mode_name, _ in modes:
            acc, _ = all_results[model_name][mode_name]
            row += f'{acc:>{col_w}.2%}'
        print(row)
    print(f'{"="*70}')

    # 스타일별 상세 (사용 가능한 최신 모델 기준)
    best_name = model_configs[-1][0]
    print(f'\n[{best_name} 스타일별 정확도]')
    mode_names = [n for n, _ in modes]
    print(f'  {"스타일":<20}' + ''.join(f'{n:>14}' for n in mode_names) + '  샘플수')
    print(f'  {"-"*70}')
    for style in STYLE_CLASSES:
        total = all_results[best_name][mode_names[0]][1][style][1]
        if total == 0:
            continue
        row = f'  {style:<20}'
        for mode_name in mode_names:
            c, t = all_results[best_name][mode_name][1][style]
            row += f'{c/t:>13.1%} '
        print(row + f' {total}')
