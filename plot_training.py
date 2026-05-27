"""학습 로그 시각화 — 에포크 단위"""
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

DATA_DIR   = Path(__file__).parent / 'data'
RESULT_DIR = Path(__file__).parent / 'result'
RESULT_DIR.mkdir(exist_ok=True)

for font in ['Malgun Gothic', 'NanumGothic', 'AppleGothic']:
    if any(f.name == font for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = font
        break
plt.rcParams['axes.unicode_minus'] = False

epoch_re = re.compile(
    r'Epoch\s+(\d+)/\d+ \| Loss: ([\d.]+) \| Train: ([\d.]+)% \| Val: ([\d.]+)%'
)

def parse_log(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                epoch, loss, train_acc, val_acc = m.groups()
                rows.append({'epoch': int(epoch), 'loss': float(loss),
                             'train_acc': float(train_acc), 'val_acc': float(val_acc)})
    return pd.DataFrame(rows)

orig = parse_log(RESULT_DIR / 'train_log.txt')
bal  = parse_log(RESULT_DIR / 'train_log_balanced.txt')

def annotate(ax, x, y, color, offset=(0, 7), fmt='{:.1f}%'):
    for xi, yi in zip(x, y):
        ax.annotate(fmt.format(yi), (xi, yi),
                    textcoords='offset points', xytext=offset,
                    fontsize=7.5, ha='center', color=color)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('기존 모델 vs 균형 모델 학습 분석', fontsize=15, fontweight='bold')

# 1. Val Accuracy
ax = axes[0, 0]
ax.plot(orig['epoch'], orig['val_acc'], 'b-o', linewidth=2, markersize=7, label='기존모델 (967,806장)')
ax.plot(bal['epoch'],  bal['val_acc'],  'r-o', linewidth=2, markersize=7, label='균형모델 (101,432장)')
annotate(ax, orig['epoch'], orig['val_acc'], 'blue', offset=(0, 7))
annotate(ax, bal['epoch'],  bal['val_acc'],  'red',  offset=(0, -14))
ax.set_title('검증 정확도 (Val Accuracy)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(range(1, 11))

# 2. Train Accuracy
ax = axes[0, 1]
ax.plot(orig['epoch'], orig['train_acc'], 'b-o', linewidth=2, markersize=7, label='기존모델')
ax.plot(bal['epoch'],  bal['train_acc'],  'r-o', linewidth=2, markersize=7, label='균형모델')
annotate(ax, orig['epoch'], orig['train_acc'], 'blue', offset=(0, 7))
annotate(ax, bal['epoch'],  bal['train_acc'],  'red',  offset=(0, -14))
ax.set_title('학습 정확도 (Train Accuracy)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(range(1, 11))

# 3. Loss
ax = axes[1, 0]
ax.plot(orig['epoch'], orig['loss'], 'b-o', linewidth=2, markersize=7, label='기존모델')
ax.plot(bal['epoch'],  bal['loss'],  'r-o', linewidth=2, markersize=7, label='균형모델')
annotate(ax, orig['epoch'], orig['loss'], 'blue', offset=(0, 7),  fmt='{:.3f}')
annotate(ax, bal['epoch'],  bal['loss'],  'red',  offset=(0, -14), fmt='{:.3f}')
ax.set_title('Loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(range(1, 11))

# 4. 과적합 Gap
ax = axes[1, 1]
orig_gap = orig['train_acc'] - orig['val_acc']
bal_gap  = bal['train_acc']  - bal['val_acc']
ax.plot(orig['epoch'], orig_gap, 'b-o', linewidth=2, markersize=7, label='기존모델')
ax.plot(bal['epoch'],  bal_gap,  'r-o', linewidth=2, markersize=7, label='균형모델')
annotate(ax, orig['epoch'], orig_gap, 'blue', offset=(0, 7))
annotate(ax, bal['epoch'],  bal_gap,  'red',  offset=(0, -14))
ax.fill_between(orig['epoch'], orig_gap, 0, alpha=0.08, color='blue')
ax.fill_between(bal['epoch'],  bal_gap,  0, alpha=0.08, color='red')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_title('과적합 Gap (Train - Val Accuracy)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Gap (%)')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(range(1, 11))

plt.tight_layout()
out = RESULT_DIR / 'training_comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'저장 완료: {out}')
plt.show()
