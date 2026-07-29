# 项目交接文档 — Qwen3-TTS 个性化语音合成

> 生成日期: 2026-07-29 | 本对话完整工作总结

---

## 1. 项目概述

**课题**: 讯飞×中科大 AI 英才班暑期实训营 — 智能语音应用与实践  
**任务**: 基于 Qwen3-TTS-12Hz-1.7B-Base 完成个性化语音合成，从数据采集到 SFT 微调到评测  
**机器**: MacBook Pro M4 Pro 48GB (本地开发) + NVIDIA A800 80GB (学校服务器 GPU 训练)

---

## 2. 环境搭建

### 2.1 Conda 环境

```bash
conda activate tts        # 专用环境
Python: 3.12.13
PyTorch: 2.13.0 (MPS, bfloat16 可用)
transformers: 4.57.3     # 官方指定版本，不是 5.x
```

环境位置: `/opt/homebrew/Caskroom/miniconda/base/envs/tts`

### 2.2 关键依赖

| 包 | 版本 | 用途 |
|------|------|------|
| qwen-tts | 0.1.1 (本地源码可编辑安装) | 核心库 |
| funasr | 1.3.26 | ASR (Paraformer) + SIM (ERes2NetV2) |
| matplotlib | 3.11.1 | 图表输出 |
| soundfile, librosa | — | 音频处理 |

### 2.3 为 macOS 做的源码修改

| 文件 | 修改内容 | 原因 |
|------|------|------|
| `qwen_tts/core/__init__.py` | 25Hz Tokenizer 改为 try/except 懒加载 | Mac 无 sox 库 |
| `qwen_tts/inference/qwen3_tts_tokenizer.py` | V1 模型注册加 None 保护 | 同上 |
| `qwen_tts/inference/qwen3_tts_model.py` | 移除 `fix_mistral_regex=True` 参数 | 该参数触发 HF 网络调用，导致离线加载失败 |
| `qwen_tts/inference/qwen3_tts_model.py` | 移除 `fix_mistral_regex` | 同上 (两个位置) |
| `pyproject.toml` | 解除 transformers/accelerate 版本锁定，移除 sox 依赖 | MPS 兼容 |
| `evaluation/run_sim.py` | funasr/matplotlib 懒加载 | --help/--dry-run 不需要安装依赖 |

### 2.4 离线模型

模型来自桌面压缩包，解压到 HuggingFace 缓存目录：
- `qwen3-base-model.tar.gz` (3.4GB) → `~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/`
- `qwen3-tts-models.tar.gz` (576MB) → `~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/`

**离线加载方法**:
```python
local_path = '/Users/wlf/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3'
model = Qwen3TTSModel.from_pretrained(local_path, ...)
```
**注意**: `tokenizer_config.json` 中的 `vocab_file` 和 `merges_file` 已被改为绝对路径。

---

## 3. 数据采集与处理

### 3.1 两轮录音数据

| | R1 (自拟文本) | R2 (标贝语料) |
|------|------|------|
| 原始文件 | `data/new/` (100 m4a) | `data/new_v2/` (100 m4a) |
| 转换产物 | `data/wavs/` (100 wav, 18MB) | `data/wavs_v2/` (100 wav, 28MB) |
| 总时长 | 6.3 min | 10.0 min |
| 文本来源 | 自拟 (日常/校园/AI) | 标贝 BZNSYP 分层抽样 |
| Git 追踪 | ✅ | ✅ |

### 3.2 数据处理 Pipeline (Phase 1-4)

全部脚本在 `data/scripts/` 下，均支持 `v1`/`v2` 版本切换：

```
data/scripts/
├── convert_m4a_to_wav.py    # m4a → 24kHz 16bit mono wav + JSONL
├── preprocess.py             # Phase 1: 静音裁剪 + 响度归一化 → wavs_clean/
├── quality_check.py          # Phase 2: ASR Paraformer CER 打分 → reports/
├── quality_check_fast.py     # Phase 2 alt: 信号特征打分 (无需网络)
├── augment.py                # Phase 3: 语速±10% + 音高±50Hz → wavs_augmented/
├── build_experiments.py      # Phase 4: 构建 14 组实验 JSONL
└── select_biaobei_100.py    # 标贝 10,000 句中选 100 句
```

**用法**: `python preprocess.py v2` 处理 R2，默认 `v1`。

### 3.3 处理后的数据目录

```
data/
├── new/                    100 m4a (R1 原始) — Git 追踪
├── new_v2/                 100 m4a (R2 原始) — Git 追踪
├── wavs/                   100 wav (R1 转换) — Git 追踪
├── wavs_v2/                100 wav (R2 转换) — Git 追踪
├── wavs_clean/             100 wav (R1 预处理) — .gitignore
├── wavs_v2_clean/          100 wav (R2 预处理) — .gitignore
├── wavs_augmented/         400 wav (R1 增强) — .gitignore
├── wavs_v2_augmented/      400 wav (R2 增强) — .gitignore
├── metadata/               文本 + 元数据 — Git 追踪
├── reports/                质检/预处理/增强报告 — .gitignore
├── experiments/            14 组实验 JSONL — .gitignore
└── scripts/                7 个脚本 — Git 追踪
```

### 3.4 ASR 质检结果

| | A 级 | B 级 | C 级 | CER 均值 | CER 中位数 |
|------|:--:|:--:|:--:|:--:|:--:|
| R1 (自拟) | 98 | 1 | 1 | **0.54%** | 0.0% |
| R2 (标贝) | 82 | 17 | 1 | **3.98%** | 0.0% |

ASR 模型: Paraformer-large (901MB)，本地路径 `/Users/wlf/.cache/modelscope/models/iic--speech_paraformer-vad-punc-zh/`

---

## 4. 训练与评测

### 4.1 GPU 训练日志

全部在 `log/` 目录下：
```
log/
├── all_experiments.log          # 第一批 12 组实验 (r1_00~r2_12, bs=2, lr=2e-6, 5 epochs)
├── remaining_experiments.log    # 第二批 4 组 (r2_13~r2_15, mixed_01~02)
├── r1_aug50_checkpointv2_v3.log # r1_04_aug50 三组对比训练
├── all_infer.log                # 推理日志
└── infer_abs.log                # 推理 abs 日志
```

### 4.2 14 组实验矩阵

| 实验组 | 训练条数 | 数据 | 策略 |
|------|:--:|------|------|
| r1_00_baseline | 90 | R1 原始 | 基线 |
| r1_01_clean30 | 30 | R1 clean Top-30 | 少而精 |
| r1_02_clean50 | 50 | R1 clean Top-50 | 质量-数量平衡 |
| r1_03_clean80 | 80 | R1 clean Top-80 | 接近全量 |
| **r1_04_aug50** | **410** | R1 clean50 + 增强 | **★ SFT 最佳** |
| r1_05_full_clean | 89 | R1 A/B 全量 | 最大清洗后 |
| r2_10_baseline | 90 | R2 原始 | R2 基线 |
| r2_11_clean30 | 30 | R2 clean Top-30 | R2 少而精 |
| ... | ... | ... | ... |
| r2_15_full_clean | 89 | R2 A/B 全量 | R2 全量 |
| mixed_01_top50 | 50 | R1+R2 各 25 | 混合 (训练崩溃) |
| mixed_02_all | 178 | R1+R2 全量 | 混合 (训练崩溃) |

> **Mixed 实验崩溃原因**: R1 和 R2 用了不同的 ref_audio，ref_mels 维度不同，collate_fn 报错 `Sizes of tensors must match`。

### 4.3 推理产物

每个实验目录下有 `inference_samples/`：
```
data/experiments/r1_04_aug50/inference_samples/
├── epoch0_seen_style.wav
├── epoch0_unseen_short.wav
├── epoch0_unseen_long.wav
├── ... (5 epochs × 3 texts = 15 wavs)
```
以及 `inference_samples_v3/` (仅 epoch14 的 3 条 v3 推理)。

所有 wav 为 24000Hz mono。

### 4.4 评测脚本

```
evaluation/
├── run_sim.py              # 原始 SIM 评测 CLI
├── sim_pipeline.py         # SIM 核心流水线
├── batch_sim_eval.py       # ★ 批量 SIM 评测 (14 组 × 5 epoch)
├── batch_cer_eval.py       # ★ 批量 CER 评测 (同上)
├── infer_compare.py        # SFT 多 checkpoint 推理对比
├── infer_fair_compare.py   # ★ 公平对比推理 (用 voice_clone API)
├── gen_zero_shot.py        # 生成 zero-shot 基线
├── analyze_logs.py         # 训练日志分析
└── tests/                  # 10/10 单元测试通过
```

### 4.5 SIM 评测结果

ERes2NetV2 模型: `iic/speech_eres2netv2_sv_zh-cn_16k-common` (71.8MB, ModelScope 缓存)

| 排名 | 实验 | SIM | 说明 |
|:--:|------|:---:|------|
| 🥇 | **zero_shot (3秒克隆)** | **0.7301** | 无微调，直接引用 |
| 🥈 | r1_04_aug50 v1 e4 | 0.6972 | SFT 最佳 (bs=2, lr=2e-6, 5ep) |
| 🥉 | r1_05_full_clean | 0.6883 | |
| 4 | mixed_02_all | 0.6865 | |
| ... | ... | ... | |
| 15 | r2_10_baseline | 0.6151 | R2 最差 |
| — | r1_04_aug50 v3 e14 | 0.6876 | v3 过拟合, 比 v1 差 |

**关键发现**: 自拟文本 SIM 比标贝高 8.9%。数据贴近个人表达习惯比文本多样性更重要。

### 4.6 CER 评测结果

**全部 14 组 × 15 条语音 CER = 0.0%** — 所有模型内容准确度完美，无吞字/复读。

### 4.7 训练超参对比 (三组 v1/v2/v3)

| | v1 (最佳) | v2 | v3 |
|------|:---:|:---:|:---:|
| batch_size | 2 | 2 | 4 |
| lr | 2e-6 | 2e-6 | 5e-6 |
| epochs | 5 | 15 | 15 |
| 最终 loss | 6.08 | 5.73 | 5.27 |
| SIM | **0.6972** | — | 0.6876 |
| 结论 | **最优** | loss 停滞(e5后) | 过拟合 |

**核心发现**: 
- 5 epoch 之后 loss 停滞(~6.3)，继续训练无益
- lr=5e-6 导致过拟合: loss 更低但 SIM 反而下降
- **最佳配置: bs=2, lr=2e-6, epochs=5**

### 4.8 公平对比方案 (未执行)

写好了 `evaluation/infer_fair_compare.py`。核心改动:
```python
# 之前 (不公平): SFT 用 generate_custom_voice(speaker="xxx")
# 公平版: SFT 用 generate_voice_clone(ref_audio=..., ref_text=...)
# 这样 SFT 和 zero-shot 用同一个 API、同一个参考音频，公平对比
```

需要把 checkpoint 传回 Mac 或直接在 GPU 上跑此脚本生成公平版音频，再传回 Mac 做 SIM 评测。

---

## 5. 报告文件

| 文件 | 说明 |
|------|------|
| `数据创新优化方案.md` | 完整实验方案 (含双轮数据对比、14 组实验矩阵) |
| `data/reports/final_evaluation_report.md` | 最终评测报告 (含 zero-shot + SFT 排名) |
| `data/reports/sim_summary.csv` | SIM 汇总 |
| `data/reports/cer_summary.csv` | CER 汇总 |
| `data/reports/sim_comparison_final.png` | SIM 对比图 (英文标签，Mac 无中文字体) |

---

## 6. Git 管理

仓库根目录: `/Users/wlf/Desktop/讯飞实训营/智能语音课题资料包/` (即本目录)

`.gitignore` 已配置:
- **追踪**: 源码(.py, .sh)、原始录音(m4a in new/)、转换 wav(wavs/, wavs_v2/)、文本(metadata/)、配置
- **忽略**: 预处理产物(wavs_clean/)、增强产物(wavs_augmented/)、实验分组(experiments/)、报告(reports/)、checkpoint、模型缓存

---

## 7. pipeline.sh 用法

```bash
conda activate tts
cd 基线代码/Qwen3-TTS-main

bash pipeline.sh quick-check          # 环境检查
bash pipeline.sh all                   # Mac 本地 Phase 1-4 (处理 v1+v2)
bash pipeline.sh phase1 v2             # 单跑 v2 预处理
bash pipeline.sh phase5 r1_04_aug50    # GPU: 提取 audio_codes
bash pipeline.sh phase6 r1_04_aug50    # GPU: SFT 训练
bash pipeline.sh phase7 r1_04_aug50    # Mac: 推理对比
bash pipeline.sh phase8                # Mac: SIM 评测
```

---

## 8. 后续可做的工作

1. **公平对比评测**: 在 GPU 上跑 `infer_fair_compare.py`，传回 `inference_fair/` 到 Mac 做 SIM
2. **将 v1 e4 checkpoint 传回 Mac**: 用公平 API 生成，对比 zero-shot SIM
3. **增加训练数据量**: 50 条独特文本是信息瓶颈，可尝试 100+ 条不同文本
4. **LoRA 替代全量 SFT**: 当前全参数微调在小数据上容易过拟合
5. **完善报告**: 补 SIM 对比图、CER 分析、主观 MOS 评测

---

## 9. 关键路径速查

```bash
# 项目根目录
cd "/Users/wlf/Desktop/讯飞实训营/智能语音课题资料包/基线代码/Qwen3-TTS-main"

# 环境
conda activate tts

# 数据脚本
ls data/scripts/

# 实验 JSONL
ls data/experiments/

# 评测脚本  
ls evaluation/

# 训练日志
ls log/

# 评测报告
ls data/reports/

# 离线模型
ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/
ls ~/.cache/modelscope/models/
```
