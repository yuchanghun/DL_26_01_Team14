"""세그멘테이션 모델 학습 — U-Net + ResNet18 백본"""
import os, json, time, warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

# ── 설정 ──────────────────────────────────────────────────
TRAIN_IMG_ROOT   = Path(r'C:\딥러닝데이터\의류통합데이터\Training\01.원천데이터')
TRAIN_LABEL_ROOT = Path(r'C:\딥러닝데이터\의류통합데이터\Training\02.라벨링데이터')
VAL_IMG_ROOT     = Path(r'C:\딥러닝데이터\의류통합데이터\Validation\01.원천데이터')
VAL_LABEL_ROOT   = Path(r'C:\딥러닝데이터\의류통합데이터\Validation\02.라벨링데이터')
SAVE_DIR         = Path(__file__).parent / 'data'
LOG_FILE         = SAVE_DIR / 'seg_train_log.txt'
CKPT_PATH        = SAVE_DIR / 'seg_checkpoint.pth'
MODEL_PATH       = SAVE_DIR / 'model_seg.pth'

IMG_SIZE    = 256
BATCH_SIZE  = 16
EPOCHS      = 10
LR          = 1e-4
NUM_WORKERS = 2

# ── 로그 ──────────────────────────────────────────────────
def log(msg):
    ts   = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# ── 데이터셋 ───────────────────────────────────────────────
class SegDataset(Dataset):
    def __init__(self, img_root, label_root, img_size=IMG_SIZE, augment=False):
        self.img_root   = img_root
        self.img_size   = img_size
        self.augment    = augment
        self.samples    = []
        self._build_index(label_root)

    def _build_index(self, label_root):
        # 이미지 인덱스: 파일명(확장자 제외) → 경로
        img_index = {}
        for p in self.img_root.rglob('*.jpg'):
            img_index[p.stem] = p

        # JSON과 이미지 매칭
        json_files = list(label_root.rglob('*.json'))
        log(f'  JSON {len(json_files):,}개 인덱싱 중...')

        def parse(jp):
            try:
                with open(jp, encoding='utf-8') as f:
                    d = json.load(f)
                ann = d['annotation']
                if not ann:
                    return None
                pts = ann[0]['annotation_point']
                w   = d['dataset']['dataset.width']
                h   = d['dataset']['dataset.height']
                stem = Path(d['dataset']['dataset.name']).stem
                img_path = img_index.get(stem)
                if img_path is None:
                    return None
                return (str(img_path), pts, w, h)
            except:
                return None

        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(parse, json_files))
        self.samples = [r for r in results if r is not None]
        log(f'  유효 샘플: {len(self.samples):,}개')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pts, w, h = self.samples[idx]

        img  = Image.open(img_path).convert('RGB')
        mask = self._make_mask(pts, w, h)

        img  = TF.resize(img,  [self.img_size, self.img_size])
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        if self.augment and torch.rand(1) > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        img_t  = TF.to_tensor(img)
        img_t  = TF.normalize(img_t, [0.485,0.456,0.406], [0.229,0.224,0.225])
        mask_t = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        return img_t, mask_t

    def _make_mask(self, points, width, height):
        pts  = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
        mask = Image.new('L', (width, height), 0)
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        return mask


# ── 메인 ──────────────────────────────────────────────────
if __name__ == '__main__':
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f'\n{"="*50}\n세그멘테이션 학습 시작: {datetime.now()}\n{"="*50}\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'디바이스: {device}')

    log('Training 데이터셋 로드...')
    train_ds = SegDataset(TRAIN_IMG_ROOT, TRAIN_LABEL_ROOT, augment=True)
    log('Validation 데이터셋 로드...')
    val_ds   = SegDataset(VAL_IMG_ROOT, VAL_LABEL_ROOT, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # U-Net + ResNet18 백본
    model = smp.Unet(
        encoder_name='resnet18',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = smp.losses.DiceLoss(mode='binary')
    scaler    = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    start_epoch = 1
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        log(f'체크포인트 로드: epoch {ckpt["epoch"]}부터 이어서 학습')

    log(f'학습 시작 | 에포크: {start_epoch}~{EPOCHS} | 배치: {BATCH_SIZE}')

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_loss, total_iou = 0.0, 0.0

        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                preds = model(imgs)
                loss  = criterion(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if (i + 1) % 500 == 0:
                log(f'  Epoch {epoch} | {i+1}/{len(train_loader)} | Loss: {total_loss/(i+1):.4f}')

        # 검증
        model.eval()
        val_iou = 0.0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds  = torch.sigmoid(model(imgs)) > 0.5
                inter  = (preds & masks.bool()).float().sum((1,2,3))
                union  = (preds | masks.bool()).float().sum((1,2,3))
                val_iou += (inter / (union + 1e-6)).mean().item()
        val_iou /= len(val_loader)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scheduler.step()

        elapsed = time.time() - epoch_start
        log(f'Epoch {epoch:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val IoU: {val_iou:.4f} | {elapsed/60:.1f}분')

        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        }, CKPT_PATH)
        torch.save(model.state_dict(), MODEL_PATH)

    log('학습 완료!')
