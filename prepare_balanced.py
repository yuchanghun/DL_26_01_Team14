"""HDD K-Fashion 데이터에서 클래스당 N장 선별해서 SSD로 복사"""
import json, shutil, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

SRC_TRAIN = Path(r'D:\백업폴더\KFashion\Training')
SRC_VAL   = Path(r'D:\백업폴더\KFashion\Validation')
DST_TRAIN = Path(r'C:\딥러닝데이터\KFashion_balanced\Training')
DST_VAL   = Path(r'C:\딥러닝데이터\KFashion_balanced\Validation')

MAX_TRAIN = 5000
MAX_VAL   = 1000
RANDOM_SEED = 42

from kfashion_dataset import STYLE_CLASSES

def build_json_index(label_base):
    """JSON 파일을 stem(이미지ID) → 경로로 인덱싱"""
    return {jp.stem: jp for jp in label_base.rglob('*.json')}

def copy_balanced(src_root, dst_root, max_per_class):
    print(f'\n[{src_root.name}] 처리 중...')
    random.seed(RANDOM_SEED)

    # 1. JSON 인덱스 (나중에 선택된 것만 찾기 위해)
    label_base = src_root / '라벨링데이터'
    print('  JSON 인덱싱 중...')
    json_index = build_json_index(label_base)
    print(f'  JSON {len(json_index):,}개 인덱싱 완료')

    # 2. 이미지 폴더에서 스타일별 선택
    img_roots = [p for i in range(1, 10)
                 if (p := src_root / f'원천데이터_{i}').exists()]
    if not img_roots:
        p = src_root / '원천데이터'
        if p.exists():
            img_roots = [p]

    style_images = {s: [] for s in STYLE_CLASSES}
    for img_root in img_roots:
        for style_dir in img_root.iterdir():
            if style_dir.name in style_images:
                style_images[style_dir.name].extend(list(style_dir.glob('*.jpg')))

    print('\n  스타일별 선택:')
    selected = []
    for style in STYLE_CLASSES:
        imgs = style_images[style]
        chosen = random.sample(imgs, min(max_per_class, len(imgs)))
        selected.extend([(img, style) for img in chosen])
        print(f'    {style:20s}: {len(imgs):6,}장 → {len(chosen):,}장')

    print(f'\n  총 {len(selected):,}장 복사 중...')

    dst_img_dir   = dst_root / '원천데이터'
    dst_label_dir = dst_root / '라벨링데이터'

    def copy_one(item):
        img_path, style = item
        # 이미지 복사
        dst_img = dst_img_dir / style / img_path.name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst_img)
        # JSON 복사 (stem으로 찾기)
        json_path = json_index.get(img_path.stem)
        if json_path:
            dst_json = dst_label_dir / style / json_path.name
            dst_json.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(json_path, dst_json)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(copy_one, selected))
    print('  복사 완료!')

if __name__ == '__main__':
    copy_balanced(SRC_TRAIN, DST_TRAIN, MAX_TRAIN)
    copy_balanced(SRC_VAL,   DST_VAL,   MAX_VAL)
    print('\n전체 완료!')
