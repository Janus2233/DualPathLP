# DualPathLP

论文《基于双路线图神经网络的动态网络链路预测》的模型与数据集代码。本仓库包含复现论文主表（表 3）与消融实验所需的全部模型代码、数据层（预处理 / 切分 / 负采样 / 指标）以及四个公开数据集的原始文件。

## 仓库结构

```
├── src/
│   ├── models/          # 模型：节点路线、子图路线、门控融合、时间编码、路线校准
│   │   ├── dual_route.py        # 双路线门控融合
│   │   ├── node_route.py        # 节点路线（GRU 记忆 + 时序图注意力）
│   │   ├── subgraph_route.py    # 子图路线（DRNL 标记 + 加权 GCN + 双端乘性读出）
│   │   ├── time_encoding.py     # 余弦时间编码
│   │   └── route_calibration.py # 路线分数校准（可选）
│   ├── graph/           # 因果封闭子图构造、候选对因果特征
│   └── train/
│       ├── m1_dual_route.py     # 主训练/评测入口（融合与单路线消融）
│       └── m0_subgraph_only.py  # 仅子图路线入口
├── data/                # 数据层：加载、预处理、时间切分、归纳留出、负采样、AP/AUC
│   └── checks/          # 数据层自检（完整性/切分/负采样/指标/预处理）
├── dataset/             # 四个数据集原始文件（SNAP 公开数据）
└── pyproject.toml       # 数据层包配置（dglp-data）
```

## 环境安装

Python ≥ 3.11。

```bash
pip install -r requirements.txt
```

- GPU 可选：论文实验在单块 6 GB 显存 GPU 上完成；CPU 可运行但更慢（`--device cpu`）。
- 需要 GPU 时请先按 [pytorch.org](https://pytorch.org) 指引安装与驱动匹配的 CUDA 版 torch，再安装 `torch_geometric`（≥ 2.6 的 GCNConv 使用原生 torch scatter，无需另装 torch_scatter/torch_sparse）。
- 论文实验环境（conda 环境实测）：Python 3.13、torch 2.9.0+cu128、torch_geometric 2.7.0、numpy 2.4.4、pandas 3.0.2、scikit-learn 1.9.0。

## 数据集

`dataset/` 已包含四个公开连续时间动态图的原始文件（来源：SNAP）：

| 注册名 | 文件 | 说明 |
|---|---|---|
| `college_msg` | CollegeMsg.txt | 加州大学欧文分校学生在线消息 |
| `email_eu_core` | email-Eu-core-temporal.txt | 欧洲研究机构内部邮件 |
| `reddit_hyperlinks` | soc-redditHyperlinks-body.txt | 跨子版块超链接 |
| `sx_mathoverflow` | sx-mathoverflow.txt | Stack Exchange 数学问答 |

文件 SHA256 与行数登记在 `data/config.py`。首次运行会自动预处理并缓存到 `runs/data/`。评测协议：按时间 70%/15%/15% 切分（分位数 0.70/0.85），另按种子 2020 留出 10% 归纳新节点并将其全部边移出训练段；训练 1:1 端点替换负采样，评测用全图固定负样本；每批 200 正 + 200 负，AP/AUC 批内计算后跨批平均。四个数据集的事件与节点属性均为 172 维全零。

数据层自检：

```bash
python -m data.checks.run_all
```

## 复现主表（表 3）

旗舰配置（CollegeMsg，种子 42，确定性模式）：

```bash
python -m src.train.m1_dual_route \
  --dataset college_msg --route fused --pair-features --input struct \
  --hop 3 --window-frac 0 --sub-layers 3 --lr 0.005 --epochs 20 \
  --seed 42 --deterministic --tag repro_college_s42
```

其余三个数据集仅子图邻域与训练日程不同（其余超参数与上式一致，均为默认值）：

```bash
python -m src.train.m1_dual_route --dataset email_eu_core    --route fused --pair-features --hop 2 --window-frac 0.05 --sub-layers 3 --lr 0.003 --epochs 15 --seed 42 --deterministic --tag repro_email_s42
python -m src.train.m1_dual_route --dataset reddit_hyperlinks --route fused --pair-features --hop 2 --window-frac 0.05 --sub-layers 3 --lr 0.003 --epochs 15 --seed 42 --deterministic --tag repro_reddit_s42
python -m src.train.m1_dual_route --dataset sx_mathoverflow   --route fused --pair-features --hop 2 --window-frac 0.05 --sub-layers 3 --lr 0.003 --epochs 15 --seed 42 --deterministic --tag repro_sx_s42
```

- 八种子结果：`--seed` 依次取 42–49，对 `test.fused.average_precision` 取均值±样本标准差。
- 单路线对照（表 4）：其余参数不变，`--route node` 或 `--route sub`。
- 结果写入 `runs/<tag>/summary.json`（`test.fused / test.node / test.sub` 各含 `average_precision` 与 `roc_auc`）；加 `--gate-stats` 另输出门控统计 `gate_stats.json`。
- CPU 运行加 `--device cpu`。

预期主表 AP（八种子均值±样本标准差）：

| 数据集 | DualPathLP（门控融合） |
|---|---|
| CollegeMsg | 0.9657 ± 0.0006 |
| email-Eu-core | 0.9858 ± 0.0001 |
| RedditHyperlinks | 0.9753 ± 0.0087 |
| sx-mathoverflow | 0.9813 ± 0.0046 |

## 复现消融与对照实验

在旗舰命令基础上追加开关：

| 实验 | 开关 |
|---|---|
| 因果统计量 → 全零输入（表 5） | `--input zeros` |
| 去除候选对因果特征（表 5） | 去掉 `--pair-features` |
| 候选对特征逐维消融 | `--pair-dims aa,jac` 等任意组合 |
| 静默移出门控上下文（表 5） | `--ctx-no-dt` |
| 静默加入节点解码器（表 5） | 默认即含；对照用 `--no-decoder-dt` |
| 常数门控对照（表 5 / 图 4b） | `--const-gate` |
| 关闭子图边时间门控（§4.4.2） | `--no-edge-gate` |
| 跨路线一致性正则（§4.4.2） | `--lambda-cons 0.05`（默认 0 关闭） |
| 严格负采样审计（表 6） | `--strict-negatives` |

## 与论文一致的实现口径（默认保持，勿随意更改）

以下三点为论文第 5 章如实报告的冻结口径，默认标志位与之对齐；改动会改变全部已冻结结果：

1. 节点路线 GRU 记忆更新在无梯度路径下执行（606,316 报告参数中 318,888 个不参与训练）；修复标志 `--memory-grad onestep` 默认关。
2. 门控上下文前两维（子图规模、边节点比）由批级量填充；修复标志 `--ctx-per-candidate` 默认关。
3. 子图边表对每条交互只登记一条有向边，图卷积沿该方向传递消息。

## 引用

《基于双路线图神经网络的动态网络链路预测》（DualPathLP）。