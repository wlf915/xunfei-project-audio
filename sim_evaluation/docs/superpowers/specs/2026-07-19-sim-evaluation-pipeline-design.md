# 自动化 SIM 评测流水线设计

## 目标

在不修改 Qwen3-TTS 核心代码的前提下，对已经生成的 zero-shot 与 SFT 合成音频进行批量说话人相似度（SIM）评测，并生成可用于报告的表格与图像。

## 输入与数据隔离

- `evaluation/data/enrollment/`：3--5 条独立真人参考录音，只用于表示目标说话人，不参与 SFT 训练。
- `evaluation/data/generated/zero_shot/`：基线 zero-shot 对测试文本生成的 WAV。
- `evaluation/data/generated/sft/`：SFT 模型对同一批测试文本生成的 WAV。
- 两个生成目录内的同名文件必须对应同一条测试文本，例如 `test_001.wav`。

训练音频和 zero-shot 推理提示音频不属于本流水线的评测输入，避免训练泄漏和提示音频复用造成的结论偏差。

## 计算方法

使用 FunASR 调用中文 ERes2NetV2 说话人验证模型 `iic/speech_eres2netv2_sv_zh-cn_16k-common`。对每条音频提取 `spk_embedding`，再计算每一条合成音频与全部 enrollment 真人录音的余弦相似度；该样本的 SIM 为所有配对得分的算术平均值。系统级 SIM 为该系统全部样本 SIM 的平均值，并报告标准差和样本数。

## 输出

- `pair_scores.csv`：每个“合成音频--真人参考音频”配对的得分。
- `sample_scores.csv`：每条合成音频跨真人参考录音的平均 SIM。
- `summary.csv`：按系统分组的均值、标准差和样本数。
- `sim_comparison.png`：zero-shot 与 SFT 的平均 SIM 和样本分布对比图。

## 错误处理与可复现性

- 支持 `--device cpu` 与 `--device cuda:0`。
- 仅接受 WAV 文件；找不到文件、目录为空、同名样本不一致时输出明确错误。
- 默认要求 zero-shot 与 SFT 样本名一致，保证对比公平。
- 所有输出写入指定 `--output-dir`，不覆盖原始音频。
