import os, sys, time, warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
import torchvision.models as models

sys.path.insert(0, str(Path(__file__).parent))
from kfashion_dataset import KFashionDataset, STYLE_CLASSES, CATEGORY_CLASSES, IMG_SIZE

# ── 설정 ──────────────────────────────────────────
TRAIN_ROOT  = r'C:\딥러닝데이터\KFashion\Training'
VAL_ROOT    = r'C:\딥러닝데이터\KFashion\Validation'
SAVE_DIR    = Path(__file__).parent / 'data'
LOG_FILE    = SAVE_DIR / 'train_log.txt'
BATCH_SIZE  = 128
EPOCHS      = 10
LR          = 1e-4
NUM_WORKERS = 2

NUM_STYLES     = len(STYLE_CLASSES)
NUM_CATEGORIES = len(CATEGORY_CLASSES)

TRAIN_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
EVAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── 로그 ──────────────────────────────────────────
def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# ── 모델 ──────────────────────────────────────────
class StyleCNN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.style_head    = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, NUM_STYLES))
        self.category_head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, NUM_CATEGORIES))

    def forward(self, x):
        feat = self.backbone(x)
        return self.style_head(feat), self.category_head(feat)

# ── 메인 ──────────────────────────────────────────
if __name__ == '__main__':
    SAVE_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f'\n{"="*50}\n학습 시작: {datetime.now()}\n{"="*50}\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'디바이스: {device}')

    # 데이터 로드
    if os.path.exists(TRAIN_ROOT):
        log('K-Fashion 데이터 로드 중...')
        train_ds = KFashionDataset(TRAIN_ROOT, TRAIN_TRANSFORM)
        log(f'  학습 샘플: {len(train_ds):,}개')
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True,
                                  persistent_workers=True)

        val_loader = None
        if os.path.exists(VAL_ROOT):
            val_ds = KFashionDataset(VAL_ROOT, EVAL_TRANSFORM)
            log(f'  검증 샘플: {len(val_ds):,}개')
            if len(val_ds) > 0:
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                        num_workers=NUM_WORKERS, pin_memory=True,
                                        persistent_workers=True)
    else:
        log('실제 데이터 없음 → 더미 데이터')
        X = torch.randn(500, 3, IMG_SIZE, IMG_SIZE)
        ys = torch.randint(0, NUM_STYLES, (500,))
        yc = torch.randint(0, NUM_CATEGORIES, (500,))
        train_ds = TensorDataset(X, ys, yc)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = None

    # 이전 체크포인트 확인
    ckpt_path = SAVE_DIR / 'model_a_checkpoint.pth'
    start_epoch = 1
    model = StyleCNN(pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        log(f'체크포인트 로드: epoch {ckpt["epoch"]}부터 이어서 학습')

    criterion = nn.CrossEntropyLoss()
    log(f'학습 시작 | 에포크: {start_epoch}~{EPOCHS} | 배치: {BATCH_SIZE} | Mixed Precision: {device.type=="cuda"}')

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_loss, correct, batch_count = 0.0, 0, 0

        for i, batch in enumerate(train_loader):
            X, ys, yc = [b.to(device, non_blocking=True) for b in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                out_s, out_c = model(X)
                loss = criterion(out_s, ys) + 0.3 * criterion(out_c, yc)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            correct    += (out_s.argmax(1) == ys).sum().item()
            batch_count += 1

            # 500배치마다 진행상황 로그
            if (i + 1) % 500 == 0:
                acc = correct / ((i + 1) * BATCH_SIZE)
                log(f'  Epoch {epoch} | {i+1}/{len(train_loader)} batches | Loss: {total_loss/batch_count:.4f} | Acc: {acc:.2%}')

        train_acc = correct / len(train_ds)
        elapsed = time.time() - epoch_start

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scheduler.step()

        # 검증
        val_info = ''
        if val_loader:
            model.eval()
            val_correct = 0
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                for batch in val_loader:
                    X, ys, yc = [b.to(device, non_blocking=True) for b in batch]
                    out_s, _ = model(X)
                    val_correct += (out_s.argmax(1) == ys).sum().item()
            val_acc = val_correct / len(val_ds)
            val_info = f' | Val: {val_acc:.2%}'

        log(f'Epoch {epoch:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train: {train_acc:.2%}{val_info} | LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed/60:.1f}분')

        # 에포크마다 체크포인트 저장
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, ckpt_path)

        # 최종 모델도 저장
        torch.save(model.state_dict(), SAVE_DIR / 'model_a_cnn.pth')

    log('학습 완료!')
