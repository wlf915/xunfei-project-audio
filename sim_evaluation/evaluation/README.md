# 自动化 SIM 评测流水线

这个目录用于比较 Qwen3-TTS 的 **zero-shot 基线** 与 **SFT 微调模型**的说话人相似度（SIM）。它不改动 Qwen3-TTS 模型源码，只读取你已经生成好的 WAV 音频。

## 1. 数据目录

```text
evaluation/
├─ data/
│  ├─ enrollment/                 # 3--5 条独立真人参考录音
│  │  ├─ real_01.wav
│  │  ├─ real_02.wav
│  │  └─ real_03.wav
│  └─ generated/
│     ├─ zero_shot/               # zero-shot 基线输出
│     │  ├─ test_001.wav
│     │  └─ test_002.wav
│     └─ sft/                     # SFT 模型输出
│        ├─ test_001.wav
│        └─ test_002.wav
├─ run_sim.py
├─ sim_pipeline.py
└─ results/                       # 运行后自动生成
```

`zero_shot/` 和 `sft/` 中的文件名必须一一对应：例如两边的 `test_001.wav` 必须由**同一条测试文本**生成。建议准备至少 20 条从未进入 SFT 训练集的新文本，因此每个系统至少有 20 条待评测音频。

## 2. 数据隔离规则

- `enrollment/` 放未用于 SFT 训练的真人录音，用来代表真实目标说话人。
- 不要把 SFT 训练集音频放入 `enrollment/`，否则 SIM 会偏高。
- 不要把 zero-shot 推理时喂给模型的提示音频同时放入 `enrollment/`；两者应分开。
- 每条音频建议为单声道 WAV；ERes2NetV2 的中文预训练模型以 16 kHz 语音为基准，16 kHz 是最稳妥的选择。

## 3. 安装依赖

在你已经能运行 Qwen3-TTS 的服务器 Conda 环境中执行：

```bash
conda activate task
python -m pip install -r evaluation/requirements.txt
```

这里不会重新安装 PyTorch。首次真实评测时，FunASR 会从 ModelScope 下载中文 ERes2NetV2 说话人验证模型 `iic/speech_eres2netv2_sv_zh-cn_16k-common`。

## 4. 先检查数据布局

在项目根目录执行：

```bash
python evaluation/run_sim.py \
  --enrollment-dir evaluation/data/enrollment \
  --generated-dir evaluation/data/generated \
  --output-dir evaluation/results \
  --dry-run
```

看到 `dry-run 验证通过` 后，说明：真人参考录音存在、两个系统的 WAV 存在，并且 zero-shot 与 SFT 的样本名完全一致。

## 5. 运行真实 SIM 评测

在 GPU 节点上执行：

```bash
python evaluation/run_sim.py \
  --enrollment-dir evaluation/data/enrollment \
  --generated-dir evaluation/data/generated \
  --output-dir evaluation/results \
  --device cuda:0
```

若没有 GPU，可改为：

```bash
--device cpu
```

如只想得到 CSV、不生成图片，可增加 `--skip-plot`。

## 6. 评分逻辑

对每一条合成音频，流水线都会先使用 ERes2NetV2 提取说话人向量（`spk_embedding`），再与 `enrollment/` 中的每一条真人音频向量计算一次余弦相似度：

```text
一条合成音频 × 3 条真人参考录音 = 3 个配对 SIM
样本级 SIM = 3 个配对 SIM 的平均值
系统级 SIM = 该系统所有样本级 SIM 的平均值
```

因此最终比较的是：

```text
zero_shot 的系统级平均 SIM  vs  SFT 的系统级平均 SIM
```

SFT 的平均 SIM 更高，表示其合成语音的音色整体更接近你的真人录音。

## 7. 输出文件说明

运行后 `evaluation/results/` 中包含：

- `pair_scores.csv`：每一个“合成音频--真人参考录音”配对的原始 SIM。
- `sample_scores.csv`：每条合成音频对多条真人录音的平均 SIM、标准差、最小值、最大值。
- `summary.csv`：zero-shot 与 SFT 的样本数、平均 SIM、标准差、最小值、最大值。
- `sim_comparison.png`：两个系统的样本级 SIM 分布与均值对比图，可直接用于报告或 PPT。

## 8. 报告中的写法

可以写为：

> 我们构建了基于中文 ERes2NetV2 说话人验证模型的自动化客观评测流水线。对于每条合成音频，系统自动提取其说话人向量，并与多条未参与训练的目标说话人真人录音向量计算余弦相似度；以多参考平均得分作为样本级 SIM，在统一的未见测试文本上对 zero-shot 基线与 SFT 模型进行批量比较，自动输出配对得分、样本级汇总、系统级均值/标准差及可视化结果。
