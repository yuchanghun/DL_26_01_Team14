"""세그멘테이션 모델 학습 — YOLOv8n-seg"""
import json, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from ultralytics import YOLO

# ── 경로 설정 ──────────────────────────────────────────────
SRC = {
    'train': {
        'img':   Path(r'C:\딥러닝데이터\의류통합데이터\Training\01.원천데이터'),
        'label': Path(r'C:\딥러닝데이터\의류통합데이터\Training\02.라벨링데이터'),
    },
    'val': {
        'img':   Path(r'C:\딥러닝데이터\의류통합데이터\Validation\01.원천데이터'),
        'label': Path(r'C:\딥러닝데이터\의류통합데이터\Validation\02.라벨링데이터'),
    },
}
YOLO_DIR = Path(r'C:\딥러닝데이터\seg_yolo')
SAVE_DIR = Path(__file__).parent / 'data'

EPOCHS     = 10
BATCH_SIZE = 16
IMG_SIZE   = 256

# 카테고리 폴더명 매핑 (라벨 TL_ → 이미지 TS_)
def label_to_img_cat(name):
    return name.replace('TL_', 'TS_', 1)

def make_junction(src, dst):
    if not dst.exists():
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(dst), str(src)],
                       check=True, capture_output=True)

def convert_one(args):
    json_path, img_cat_dir, label_dst_dir = args
    try:
        with open(json_path, encoding='utf-8') as f:
            d = json.load(f)
        ann = d['annotation']
        if not ann:
            return False
        pts = ann[0]['annotation_point']
        w   = d['dataset']['dataset.width']
        h   = d['dataset']['dataset.height']
        stem = Path(d['dataset']['dataset.name']).stem

        img_path = img_cat_dir / (stem + '.jpg')
        if not img_path.exists():
            return False

        coords = [(pts[i], pts[i+1]) for i in range(0, len(pts) - 1, 2)]
        norm   = ' '.join(f'{x/w:.6f} {y/h:.6f}' for x, y in coords)

        (label_dst_dir / (stem + '.txt')).write_text(f'0 {norm}\n', encoding='utf-8')
        return True
    except:
        return False

def prepare_dataset():
    yaml_path = YOLO_DIR / 'dataset.yaml'
    if yaml_path.exists():
        print('변환된 데이터셋 존재, 스킵')
        return yaml_path

    for split in ['train', 'val']:
        img_src   = SRC[split]['img']
        label_src = SRC[split]['label']

        for label_cat in label_src.iterdir():
            if not label_cat.is_dir():
                continue

            img_cat_name = label_to_img_cat(label_cat.name)
            img_cat_dir  = img_src / img_cat_name

            # images 폴더에 junction
            img_link = YOLO_DIR / 'images' / split / img_cat_name
            img_link.parent.mkdir(parents=True, exist_ok=True)
            make_junction(img_cat_dir, img_link)

            # labels 폴더 생성
            label_dst = YOLO_DIR / 'labels' / split / img_cat_name
            label_dst.mkdir(parents=True, exist_ok=True)

            json_files = list(label_cat.rglob('*.json'))
            print(f'  [{split}] {label_cat.name}: {len(json_files):,}개 변환 중...')

            args = [(jp, img_cat_dir, label_dst) for jp in json_files]
            with ThreadPoolExecutor(max_workers=8) as ex:
                results = list(ex.map(convert_one, args))
            print(f'    완료: {sum(results):,}개')

    yaml_path.write_text(
        f"path: {YOLO_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['clothing']\n",
        encoding='utf-8'
    )
    print('dataset.yaml 생성 완료')
    return yaml_path


if __name__ == '__main__':
    yaml_path = prepare_dataset()

    model = YOLO('yolov8n-seg.pt')
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=str(SAVE_DIR),
        name='seg_yolo',
        exist_ok=True,
        device=0,
    )

    best = SAVE_DIR / 'seg_yolo' / 'weights' / 'best.pt'
    if best.exists():
        import shutil
        shutil.copy(best, SAVE_DIR / 'model_seg.pt')
        print(f'모델 저장 완료: {SAVE_DIR / "model_seg.pt"}')
