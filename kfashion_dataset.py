import json
import pickle
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

STYLE_CLASSES = [
    '레트로', '로맨틱', '리조트', '매니시',
    '모던', '밀리터리', '섹시', '소피스트케이티드', '스트리트',
    '스포티', '아방가르드', '오리엔탈', '웨스턴', '젠더리스',
    '컨트리', '클래식', '키치', '톰보이', '펑크',
    '페미닌', '프레피', '히피', '힙합',
]  # 23개 (기타 제거)

CATEGORY_CLASSES = ['상의', '하의', '아우터', '원피스']  # 4개 (점프수트→원피스 통합)

IMG_SIZE = 224
_CACHE_VERSION = 3

_STYLE_IDX = {s: i for i, s in enumerate(STYLE_CLASSES)}

EVAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class KFashionDataset(Dataset):
    """
    K-Fashion 멀티라벨 데이터셋
    - 스타일: 주스타일 + 서브스타일 → float32 이진 벡터 (23차원)
    - 카테고리: 상의/하의/아우터/원피스 단일라벨
    - bbox: JSON 렉트좌표로 옷 영역 크롭
    - 기타 폴더(스타일 없는 이미지) 제외
    """
    def __init__(self, data_root, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform or EVAL_TRANSFORM
        self.samples = []
        self._load()

    def _load(self):
        label_base = self.data_root / '라벨링데이터'
        cache_path = self.data_root / 'samples_cache.pkl'

        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if (cached.get('data_root') == str(self.data_root)
                    and cached.get('version') == _CACHE_VERSION):
                self.samples = cached['samples']
                print(f'  캐시 로드: {len(self.samples):,}개')
                return
            print('  캐시 재구축')

        print('  이미지 인덱스 구축 중...')
        img_index = {}
        img_roots = [p for i in range(1, 10)
                     if (p := self.data_root / f'원천데이터_{i}').exists()]
        if not img_roots:
            p = self.data_root / '원천데이터'
            if p.exists():
                img_roots = [p]
        for img_root in img_roots:
            for p in img_root.rglob('*.jpg'):
                try:
                    img_index[int(p.stem)] = p.relative_to(self.data_root)
                except ValueError:
                    pass
        print(f'  이미지: {len(img_index):,}장')

        json_files = list(label_base.rglob('*.json'))
        print(f'  JSON 파싱 중: {len(json_files):,}개 (8스레드)...')

        def parse_one(jp):
            try:
                with open(jp, encoding='utf-8') as f:
                    lbl = json.load(f)
                info   = lbl['이미지 정보']
                detail = lbl['데이터셋 정보']['데이터셋 상세설명']
                lab    = detail['라벨링']
                identifier = int(info['이미지 식별자'])
            except Exception:
                return None

            rel_path = img_index.get(identifier)
            if rel_path is None:
                return None

            # 스타일 멀티라벨 벡터
            style_list = lab.get('스타일', [{}])
            if not (style_list and isinstance(style_list[0], dict) and style_list[0]):
                return None  # 스타일 없음 → 기타 폴더, 제외

            label_vec = [0.0] * len(STYLE_CLASSES)
            main_style = style_list[0].get('스타일', '')
            sub_style  = style_list[0].get('서브스타일', '')
            valid = False
            # 스트리트+서브스타일 있으면 서브스타일로 교체 (스트리트 제거)
            if main_style == '스트리트' and sub_style in _STYLE_IDX:
                label_vec[_STYLE_IDX[sub_style]] = 1.0
                valid = True
            else:
                if main_style in _STYLE_IDX:
                    label_vec[_STYLE_IDX[main_style]] = 1.0
                    valid = True
                if sub_style in _STYLE_IDX:
                    label_vec[_STYLE_IDX[sub_style]] = 1.0
                    valid = True
            if not valid:
                return None

            # 카테고리 + bbox
            category_idx = 0
            bbox = None
            rect_data = detail.get('렉트좌표', {})
            for i, cat in enumerate(CATEGORY_CLASSES):
                items = lab.get(cat, [{}])
                if items and isinstance(items[0], dict) and items[0]:
                    category_idx = i
                    rect_items = rect_data.get(cat, [{}])
                    if rect_items and isinstance(rect_items[0], dict) and rect_items[0]:
                        r = rect_items[0]
                        x = r.get('X좌표', 0)
                        y = r.get('Y좌표', 0)
                        w = r.get('가로', 0)
                        h = r.get('세로', 0)
                        if w > 0 and h > 0:
                            bbox = (x, y, x + w, y + h)
                    break

            return (str(rel_path), label_vec, category_idx, bbox)

        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(parse_one, json_files))

        self.samples = [r for r in results if r is not None]

        with open(cache_path, 'wb') as f:
            pickle.dump({
                'data_root': str(self.data_root),
                'version': _CACHE_VERSION,
                'samples': self.samples,
            }, f)
        print(f'  완료: {len(self.samples):,}개 (캐시 저장)')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label_vec, cat_idx, bbox = self.samples[idx]
        img = Image.open(self.data_root / rel_path).convert('RGB')
        if bbox is not None:
            img = img.crop(bbox)
        return (
            self.transform(img),
            torch.tensor(label_vec, dtype=torch.float32),
            cat_idx,
        )
