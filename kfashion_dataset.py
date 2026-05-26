import json
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

STYLE_CLASSES = [
    '기타', '레트로', '로맨틱', '리조트', '매니시',
    '모던', '밀리터리', '섹시', '소피스트케이티드', '스트리트',
    '스포티', '아방가르드', '오리엔탈', '웨스턴', '젠더리스',
    '컨트리', '클래식', '키치', '톰보이', '펑크',
    '페미닌', '프레피', '히피', '힙합',
]
CATEGORY_CLASSES = ['상의', '하의', '아우터', '원피스', '점프수트']

IMG_SIZE = 224

EVAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class KFashionDataset(Dataset):
    """
    K-Fashion AI Hub 데이터셋 로더
    - 이미지 식별자(정수)로 JSON↔이미지 매칭
    - 스타일: JSON 우선, 비어있으면 폴더명 사용
    - 캐시에 상대경로 저장 → 데이터 이동해도 안전
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
            if cached.get('data_root') == str(self.data_root):
                self.samples = cached['samples']
                print(f'  캐시 로드: {len(self.samples):,}개')
                return
            print('  data_root 변경 감지 → 캐시 재구축')

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
                    img_index[int(p.stem)] = (p.parent.name, p.relative_to(self.data_root))
                except ValueError:
                    pass
        print(f'  이미지: {len(img_index):,}장')

        json_files = list(label_base.rglob('*.json'))
        print(f'  JSON 파싱 중: {len(json_files):,}개 (8스레드)...')

        def parse_one(jp):
            try:
                with open(jp, encoding='utf-8') as f:
                    lbl = json.load(f)
                info = lbl['이미지 정보']
                lab  = lbl['데이터셋 정보']['데이터셋 상세설명']['라벨링']
                identifier = int(info['이미지 식별자'])
            except Exception:
                return None
            result = img_index.get(identifier)
            if result is None:
                return None
            folder_name, rel_path = result
            style_list = lab.get('스타일', [{}])
            if style_list and isinstance(style_list[0], dict) and '스타일' in style_list[0]:
                style = style_list[0]['스타일']
            else:
                style = folder_name
            if style not in STYLE_CLASSES:
                return None
            category = CATEGORY_CLASSES[0]
            for cat in CATEGORY_CLASSES:
                items = lab.get(cat, [{}])
                if items and isinstance(items[0], dict) and items[0]:
                    category = cat
                    break
            return (str(rel_path), STYLE_CLASSES.index(style), CATEGORY_CLASSES.index(category))

        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(parse_one, json_files))

        self.samples = [r for r in results if r is not None]

        with open(cache_path, 'wb') as f:
            pickle.dump({'data_root': str(self.data_root), 'samples': self.samples}, f)
        print(f'  완료: {len(self.samples):,}개 (캐시 저장)')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, s, c = self.samples[idx]
        img = Image.open(self.data_root / rel_path).convert('RGB')
        return self.transform(img), s, c
