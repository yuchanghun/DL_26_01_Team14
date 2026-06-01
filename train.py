import os, sys, time, warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models

sys.path.insert(0, str(Path(__file__).parent))
from kfashion_dataset import KFashionDataset, STYLE_CLASSES, CATEGORY_CLASSES, IMG_SIZE

# ── 설정 ──────────────────────────────────────────
TRAIN_ROOT  = r'C:\딥러닝데이터\KFashion\Training'
VAL_ROOT    = r'C:\딥러닝데이터\KFashion\Validation'
SAVE_DIR    = Path(__file__).parent / 'model'
LOG_FILE    = Path(__file__).parent / 'result' / 'train_log_v4.txt'
MODEL_NAME  = 'model_a_v4'
BATCH_SIZE  = 64
EPOCHS      = 15
LR          = 1e-5
NUM_WORKERS = 4

NUM_STYLES     = len(STYLE_CLASSES)
NUM_CATEGORIES = len(CATEGORY_CLASSES)

TRAIN_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
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
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.style_head    = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(2048, NUM_STYLES))
        self.category_head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(2048, NUM_CATEGORIES))

    def forward(self, x):
        feat = self.backbone(x)
        return self.style_head(feat), self.category_head(feat)

# ── Top-N Recall (멀티라벨) ──────────────────────
def top_n_recall(logits, labels, n=3):
    topn = logits.topk(n, dim=1).indices
    hits = 0
    total = 0
    for i in range(len(labels)):
        pos = labels[i].nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            continue
        hits += sum(1 for p in pos if p in topn[i])
        total += len(pos)
    return hits / total if total > 0 else 0.0

# ── 메인 ──────────────────────────────────────────
if __name__ == '__main__':
    SAVE_DIR.mkdir(exist_ok=True)
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f'\n{"="*50}\n학습 시작: {datetime.now()}\n{"="*50}\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'디바이스: {device}')

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

    # pos_weight 계산 (클래스 불균형 처리)
    log('pos_weight 계산 중...')
    pos_counts = torch.zeros(NUM_STYLES)
    total = len(train_ds)
    for _, label_vec, _, _ in train_ds.samples:
        pos_counts += torch.tensor(label_vec)
    neg_counts = total - pos_counts
    pos_weight = (neg_counts / pos_counts.clamp(min=1)).clamp(max=20).to(device)
    log(f'  pos_weight 범위: {pos_weight.min():.1f} ~ {pos_weight.max():.1f}')

    # 체크포인트 확인
    ckpt_path = SAVE_DIR / f'{MODEL_NAME}_checkpoint.pth'
    start_epoch = 1
    model = StyleCNN(pretrained=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        log(f'체크포인트 로드: epoch {ckpt["epoch"]}부터 이어서 학습')
    else:
        # v3 가중치로 초기화 (LR 리셋)
        prev_weights = SAVE_DIR / 'model_a_v3.pth'
        model.load_state_dict(torch.load(prev_weights, map_location=device, weights_only=True))
        log(f'v3 가중치 로드 완료 (LR 리셋: {LR:.0e})')

    style_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    cat_criterion   = nn.CrossEntropyLoss()
    log(f'학습 시작 | 에포크: {start_epoch}~{EPOCHS} | 배치: {BATCH_SIZE} | ResNet50 + BCE')

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        batch_count = 0

        for i, (X, ys, yc) in enumerate(train_loader):
            X  = X.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            yc = yc.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                out_s, out_c = model(X)
                loss = style_criterion(out_s, ys) + 0.3 * cat_criterion(out_c, yc)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            batch_count += 1

            if (i + 1) % 500 == 0:
                log(f'  Epoch {epoch} | {i+1}/{len(train_loader)} | Loss: {total_loss/batch_count:.4f}')

        elapsed = time.time() - epoch_start
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scheduler.step()

        # 검증
        val_info = ''
        if val_loader:
            model.eval()
            r1_sum, r3_sum, n_batches = 0.0, 0.0, 0
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                for X, ys, yc in val_loader:
                    X  = X.to(device, non_blocking=True)
                    ys = ys.to(device, non_blocking=True)
                    out_s, _ = model(X)
                    r1_sum += top_n_recall(out_s, ys, n=1)
                    r3_sum += top_n_recall(out_s, ys, n=3)
                    n_batches += 1
            r1 = r1_sum / n_batches
            r3 = r3_sum / n_batches
            val_info = f' | Val R@1: {r1:.2%} R@3: {r3:.2%}'

        log(f'Epoch {epoch:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}{val_info} | LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed/60:.1f}분')

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, ckpt_path)
        torch.save(model.state_dict(), SAVE_DIR / f'{MODEL_NAME}.pth')

    log('학습 완료!')
