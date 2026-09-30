"""
教学脚本：在代码里怎么"看见"过拟合/欠拟合 + 各种评价指标。

场景沿用"跌倒检测"：二分类，正类(跌倒)只占 ~5%，是个不平衡问题。
跑一次会得到两张图 + 一堆打印，对应面试要讲的所有概念。

运行： D:/D_conda/envs/pytorch/python.exe demo_metrics_overfit.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")    # 让 Windows 终端能打印中文
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")                       # 不弹窗，直接存文件
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 图里能显示中文
plt.rcParams["axes.unicode_minus"] = False
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, classification_report,
                             roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)

torch.manual_seed(0); np.random.seed(0)

# ---------------------------------------------------------------------------
# 0) 造一个不平衡的二分类数据集（跌倒=1 占 5%，正常=0 占 95%）
# ---------------------------------------------------------------------------
X, y = make_classification(n_samples=4000, n_features=20, n_informative=6,
                           weights=[0.95, 0.05], random_state=0)
# 故意只拿很小一块当训练集 -> 容易过拟合（数据少+模型大=背噪声）
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, train_size=300,
                                            stratify=y, random_state=0)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                            stratify=y_tmp, random_state=0)
to_t = lambda a: torch.tensor(a, dtype=torch.float32)
X_tr_t, X_val_t = to_t(X_tr), to_t(X_val)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1) 一个小 MLP。dropout/weight_decay 是"压方差防过拟合"的开关。
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, 1))                    # 输出 1 个 logit
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(p_drop, weight_decay, epochs=300):
    """训练并【每个 epoch 同时记录 训练集 和 验证集 的 loss/acc】。
    能不能看见过拟合，全靠同时盯这两条线。"""
    model = MLP(p_drop)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()              # 二分类交叉熵
    hist = {"tr_loss": [], "val_loss": [], "tr_acc": [], "val_acc": []}
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model(X_tr_t), y_tr_t)
        loss.backward(); opt.step()
        # ---- 每个 epoch 在两个集合上各评一次 ----
        model.eval()
        with torch.no_grad():
            for tag, Xs, ys in [("tr", X_tr_t, y_tr_t), ("val", X_val_t, y_val_t)]:
                logit = model(Xs)
                l = lossf(logit, ys).item()
                acc = ((logit > 0).float() == ys).float().mean().item()
                hist[f"{tag}_loss"].append(l)
                hist[f"{tag}_acc"].append(acc)
    return model, hist


# 两个版本：A 不正则化(会过拟合)  B 正则化(dropout+weight_decay)
model_overfit, h_of = train(p_drop=0.0, weight_decay=0.0)
model_reg,     h_rg = train(p_drop=0.3, weight_decay=1e-3)

print("="*70)
print("① 怎么看过拟合：比 训练 vs 验证 这两条线")
print("="*70)
print(f"[不正则化] 最后一轮  训练acc={h_of['tr_acc'][-1]:.3f}  验证acc={h_of['val_acc'][-1]:.3f}"
      f"   <- 训练近满分、验证差一截 = 过拟合")
print(f"[不正则化] 验证loss 最低={min(h_of['val_loss']):.3f} -> 最后={h_of['val_loss'][-1]:.3f}"
      f"   <- 验证loss先降后升 = 开始背噪声(过拟合的铁证)")
print(f"[正则化  ] 最后一轮  训练acc={h_rg['tr_acc'][-1]:.3f}  验证acc={h_rg['val_acc'][-1]:.3f}"
      f"   <- 两条线更贴近 = 泛化更好")

# ---- 图1：4 张子图，一眼看穿过拟合 ----
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for col, (h, title) in enumerate([(h_of, "NO reg (overfit)"), (h_rg, "WITH reg")]):
    ax[0, col].plot(h["tr_loss"], label="train"); ax[0, col].plot(h["val_loss"], label="val")
    ax[0, col].set_title(f"{title} - loss"); ax[0, col].legend(); ax[0, col].set_xlabel("epoch")
    ax[1, col].plot(h["tr_acc"], label="train"); ax[1, col].plot(h["val_acc"], label="val")
    ax[1, col].set_title(f"{title} - accuracy"); ax[1, col].legend(); ax[1, col].set_xlabel("epoch")
fig.suptitle("过拟合 = 训练好但验证差(左);  正则化后两线贴合(右)")
fig.tight_layout(); fig.savefig("demo_fig1_overfit.png", dpi=110)


# ---------------------------------------------------------------------------
# 2) 评价指标：拿正则化模型在【测试集】上算。
#    模型先输出概率，再和阈值比，才变成 0/1 预测。
# ---------------------------------------------------------------------------
model_reg.eval()
with torch.no_grad():
    prob = torch.sigmoid(model_reg(to_t(X_te))).numpy()   # 跌倒的概率 0~1
pred_05 = (prob >= 0.5).astype(int)                       # 默认阈值 0.5

print("\n" + "="*70)
print("② 怎么看评价指标（都从混淆矩阵来）")
print("="*70)
cm = confusion_matrix(y_te, pred_05)                      # 行=真实, 列=预测
tn, fp, fn, tp = cm.ravel()
print("混淆矩阵 [[TN FP],[FN TP]] =\n", cm)
print(f"  TN={tn}(正常判对) FP={fp}(误报) FN={fn}(漏报!) TP={tp}(抓对跌倒)")
print(f"\n阈值0.5下：")
print(f"  准确率 accuracy = {accuracy_score(y_te, pred_05):.3f}")
print(f"  精确率 precision= {precision_score(y_te, pred_05):.3f}  (报警里多少是真的, 怕误报)")
print(f"  召回率 recall   = {recall_score(y_te, pred_05):.3f}  (真跌倒抓到几成, 怕漏报)")
print(f"  F1            = {f1_score(y_te, pred_05):.3f}  (两者调和平均)")

# 准确率陷阱：一个"永远说正常"的傻模型
dummy = np.zeros_like(y_te)
print(f"\n准确率陷阱：'永远说正常'的傻模型 accuracy={accuracy_score(y_te, dummy):.3f}"
      f"  但 recall={recall_score(y_te, dummy, zero_division=0):.3f} (一个跌倒没抓到!)")

print("\nsklearn 一行出全部(每类的 P/R/F1)：")
print(classification_report(y_te, pred_05, target_names=["正常", "跌倒"], digits=3))

# ---- 阈值权衡：调阈值，看 precision/recall 此消彼长 ----
print("精确率 vs 召回率 的权衡(改阈值)：")
print(f"{'阈值':>6}{'精确率':>10}{'召回率':>10}   <- 健康岗: 阈值调低保召回(别漏报)")
for thr in [0.7, 0.5, 0.3, 0.15, 0.05]:
    p = (prob >= thr).astype(int)
    print(f"{thr:>6.2f}{precision_score(y_te,p,zero_division=0):>10.3f}"
          f"{recall_score(y_te,p,zero_division=0):>10.3f}")

# ---- 阈值无关的整体指标 ----
print(f"\n阈值无关的整体评价：")
print(f"  ROC-AUC = {roc_auc_score(y_te, prob):.3f}  (排序能力; 不平衡时偏乐观)")
print(f"  PR-AUC  = {average_precision_score(y_te, prob):.3f}  (不平衡时更诚实, 健康岗看这个)")

# ---- 图2：ROC 曲线 + PR 曲线 ----
fpr, tpr, _ = roc_curve(y_te, prob)
prec, rec, _ = precision_recall_curve(y_te, prob)
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.5))
ax2[0].plot(fpr, tpr); ax2[0].plot([0, 1], [0, 1], "--", c="gray")
ax2[0].set_xlabel("FPR(误报率)"); ax2[0].set_ylabel("TPR=召回率")
ax2[0].set_title(f"ROC  AUC={roc_auc_score(y_te,prob):.3f}")
ax2[1].plot(rec, prec)
ax2[1].set_xlabel("召回率"); ax2[1].set_ylabel("精确率")
ax2[1].set_title(f"PR曲线  AP={average_precision_score(y_te,prob):.3f}")
fig2.tight_layout(); fig2.savefig("demo_fig2_roc_pr.png", dpi=110)
print("\n两张图已存： demo_fig1_overfit.png  /  demo_fig2_roc_pr.png")
