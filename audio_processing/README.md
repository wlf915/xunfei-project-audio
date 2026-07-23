# Qwen3-TTS 单说话人语音微调实验

## 1. 项目简介

本项目完成了 Qwen3-TTS 的官方推理流程复现，以及基于个人中文语音数据的单说话人微调实验。

实验以 `Qwen3-TTS-12Hz-1.7B-Base` 为基础模型，使用 100 条个人中文语音数据完成数据检查、训练集与测试集划分、离散语音编码提取、模型训练和微调后推理。

本实验采用的是：

```text
Full-parameter Supervised Fine-Tuning
```

即全参数监督微调，不是 LoRA 微调。

实验最终成功训练 3 个 epoch，并保存 3 个 checkpoint。随后对每个 checkpoint 使用相同的 3 类测试文本进行生成，共得到 9 条对比语音。

---

## 2. 实验目标

本实验主要包含以下目标：

1. 复现 Qwen3-TTS 官方推理流程；
2. 验证 Qwen3-TTS 的 zero-shot voice cloning 能力；
3. 构建个人单说话人语音数据集；
4. 使用官方 12Hz Tokenizer 提取 `audio_codes`；
5. 对 Qwen3-TTS 1.7B Base 模型进行单说话人全参数微调；
6. 比较不同训练 epoch 的生成效果；
7. 保存实验代码、配置、日志和生成音频，形成可复现的实验记录。

---

## 3. 基础模型

本实验使用以下两个官方模型：

### 3.1 基础语音生成模型

```text
Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

用途：

* 加载预训练 Qwen3-TTS 模型；
* 进行单说话人全参数微调；
* 从微调后的 checkpoint 生成语音。

### 3.2 音频 Tokenizer

```text
Qwen/Qwen3-TTS-Tokenizer-12Hz
```

用途：

* 将训练音频转换为离散语音编码；
* 为训练 JSONL 增加 `audio_codes` 字段；
* 生成官方微调脚本需要的训练数据格式。

本仓库不直接上传基础模型和 Tokenizer，使用者需要从官方模型仓库单独下载。

---

## 4. 官方代码版本

实验基于 Qwen3-TTS 官方仓库完成。

使用的代码 commit：

```text
022e286b98fbec7e1e916cb940cdf532cd9f488e
```

主要使用的官方脚本包括：

```text
finetuning/prepare_data.py
finetuning/sft_12hz.py
```

其中：

* `prepare_data.py` 用于提取离散语音编码；
* `sft_12hz.py` 用于单说话人监督微调。

为适配当前版本的 Hugging Face Accelerate，本实验对训练脚本中的 TensorBoard 日志目录进行了少量修改。

---

## 5. 实验环境

### 5.1 操作系统

```text
Ubuntu 22.04
```

### 5.2 硬件

训练阶段使用：

```text
NVIDIA A800 80GB
GPU 数量：1
```

早期的推理复现和数据预处理曾使用 RTX 4090 24GB。

### 5.3 Python 环境

```text
Python 3.12
PyTorch 2.6.0+cu124
PyTorch CUDA Runtime 12.4
CUDA Toolkit 12.4
Transformers 4.57.3
Mixed Precision BF16
FlashAttention 2
```

### 5.4 FlashAttention 说明

原始环境曾使用：

```text
PyTorch 2.13.0+cu130
```

而服务器本地 `nvcc` 为 CUDA 12.4，导致 FlashAttention 源码编译时出现 CUDA 版本不匹配。

因此重新创建训练环境，使版本统一为：

```text
PyTorch CUDA：12.4
nvcc：12.4
```

之后针对当前 PyTorch、CUDA 和 C++ ABI 强制从源码编译 FlashAttention 2，并成功通过：

* Python 导入测试；
* CUDA Kernel 运算测试；
* Qwen3-TTS FlashAttention 2 模型加载测试。

---

## 6. 官方推理复现

在微调之前，首先完成了 Qwen3-TTS 的官方 zero-shot 推理复现。

### 6.1 官方参考音频推理

成功生成：

```text
official_zero_shot.wav
```

音频信息：

```text
编码：PCM 16-bit
采样率：24000 Hz
声道：单声道
时长：约 5.84 秒
```

### 6.2 个人参考音频推理

使用个人参考音频完成 zero-shot voice cloning，并成功生成：

```text
personal_zero_shot.wav
```

音频信息：

```text
编码：PCM 16-bit
采样率：24000 Hz
声道：单声道
时长：约 8.72 秒
```

这一步验证了基础模型、Tokenizer、CUDA 环境和推理代码均能正常工作。

---

## 7. 数据集说明

### 7.1 数据规模

本实验共使用：

```text
100 条个人中文语音
```

文件命名格式：

```text
001.wav
002.wav
...
100.wav
```

对应文本编号格式：

```text
000001
000002
...
000100
```

脚本通过整数编号映射，将六位文本 ID 对应到三位音频文件名。

### 7.2 音频格式

所有音频统一为：

```text
采样率：24000 Hz
声道：1
编码：PCM signed 16-bit little-endian
```

即：

```text
pcm_s16le, 24000 Hz, mono
```

### 7.3 数据检查

数据预检查结果：

```text
文本行数：100
WAV 文件数：100
缺失音频：0
多余音频：0
重复编号：0
无效文本行：0
格式异常：0
```

音频质量检查中仅有 `079.wav` 被自动规则标记为：

```text
possible_clipping
```

经过人工试听后没有发现明显异常，因此保留该样本，不做修改。

---

## 8. 统一参考音频

官方微调说明建议，单说话人训练中的所有样本使用同一个 `ref_audio`。

本实验从数据集中选择了一条：

* 音质清晰；
* 语速自然；
* 情绪中性；
* 时长适中；
* 无明显噪声和爆音；

的音频作为统一参考音频。

训练 JSONL 中的所有样本均引用同一条参考音频。

出于个人语音隐私考虑，本仓库不上传该参考音频。

---

## 9. 数据集划分

100 条数据按照固定规则划分为：

```text
训练集：90 条
测试集：10 条
```

划分规则为：

```text
编号能被 10 整除的样本作为测试集
```

测试集编号：

```text
000010
000020
000030
000040
000050
000060
000070
000080
000090
000100
```

其余 90 条作为训练集。

数据划分文件位于：

```text
metadata/train_ids.txt
metadata/test_ids.txt
```

采用固定规则划分，有利于后续复现实验和比较不同训练配置。

---

## 10. 原始训练 JSONL

官方数据预处理脚本要求每一行包含：

```json
{
  "audio": "/absolute/path/to/audio.wav",
  "text": "该音频对应的文本",
  "ref_audio": "/absolute/path/to/reference.wav"
}
```

示例：

```json
{
  "audio": "data/wavs/001.wav",
  "text": "这里是一条示例训练文本。",
  "ref_audio": "data/reference/ref_speaker.wav"
}
```

示例文件位于：

```text
metadata/train_raw.example.jsonl
```

真实训练 JSONL 中包含本地绝对路径，因此没有上传到仓库。

用于生成 JSONL 的脚本位于：

```text
scripts/create_finetune_jsonl.py
```

示例运行方式：

```bash
python scripts/create_finetune_jsonl.py \
  --wav-dir /path/to/wavs \
  --text-file /path/to/text.txt \
  --ref-audio /path/to/ref_speaker.wav \
  --train-output /path/to/train_raw.jsonl \
  --test-output /path/to/test_raw.jsonl
```

脚本将自动：

1. 读取文本编号和文本内容；
2. 将六位文本编号映射到三位 WAV 文件名；
3. 检查所有音频是否存在；
4. 按编号是否能被 10 整除划分训练集和测试集；
5. 输出训练和测试 JSONL。

---

## 11. 提取 audio_codes

原始 JSONL 不能直接用于训练，需要使用官方 12Hz Tokenizer 提取离散语音编码。

运行命令：

```bash
python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path /path/to/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl /path/to/train_raw.jsonl \
  --output_jsonl /path/to/train_with_codes.jsonl
```

输入文件：

```text
train_raw.jsonl
```

输出文件：

```text
train_with_codes.jsonl
```

处理完成后，每条数据增加：

```text
audio_codes
```

字段。

最终检查结果：

```text
有效 JSON 行数：90
audio_codes 为空：0
缺少字段：0
JSON 解析失败：0
```

说明 90 条训练数据全部成功完成离散语音编码提取。

由于 `train_with_codes.jsonl` 包含个人数据、本地路径和离散语音表示，本仓库不上传该文件。

---

## 12. 微调方式

本实验使用官方单说话人 SFT 脚本：

```text
finetuning/sft_12hz.py
```

微调方式为：

```text
Full-parameter SFT
```

即训练过程中更新模型的完整可训练参数。

本实验没有使用：

```text
LoRA
QLoRA
PEFT
Adapter
```

仓库中的：

```text
scripts/sft_12hz_modified.py
```

为适配当前 Accelerate 与 TensorBoard 环境后的训练脚本版本。

---

## 13. 训练配置

第一轮实验配置如下：

```yaml
experiment_name: exp01_lr2e-6_bs2_ep3

model:
  name: Qwen3-TTS-12Hz-1.7B-Base
  tokenizer: Qwen3-TTS-Tokenizer-12Hz
  fine_tuning_method: full_parameter_sft
  attention: flash_attention_2
  precision: bf16

dataset:
  total_samples: 100
  train_samples: 90
  test_samples: 10
  sample_rate: 24000
  channels: 1
  sample_format: pcm_s16le

training:
  speaker_name: student_voice
  batch_size: 2
  gradient_accumulation_steps: 4
  effective_batch_size: 8
  learning_rate: 2.0e-6
  num_epochs: 3

hardware:
  gpu: NVIDIA A800 80GB
  gpu_count: 1
```

完整配置文件位于：

```text
configs/exp01.yaml
```

### 13.1 有效 Batch Size

训练脚本中：

```text
batch_size = 2
gradient_accumulation_steps = 4
```

因此有效 batch size 为：

```text
2 × 4 = 8
```

### 13.2 学习率

```text
2e-6
```

考虑到数据集仅有 90 条训练样本，并且使用全参数微调，因此采用较小学习率，降低小数据条件下模型快速过拟合或训练不稳定的风险。

### 13.3 训练轮数

```text
3 epochs
```

3 个 epoch 作为第一轮实验，用于验证训练流程并观察不同 checkpoint 的生成变化。

---

## 14. 训练命令

训练命令如下：

```bash
python finetuning/sft_12hz.py \
  --init_model_path /path/to/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path /path/to/checkpoints/exp01_lr2e-6_bs2_ep3 \
  --train_jsonl /path/to/train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-6 \
  --num_epochs 3 \
  --speaker_name student_voice
```

训练使用：

```text
BF16 mixed precision
FlashAttention 2
```

训练阶段未发生：

```text
CUDA out of memory
Loss NaN
数据读取失败
训练进程异常退出
```

---

## 15. 训练结果

训练成功完成 3 个 epoch。

保存的 checkpoint：

```text
checkpoint-epoch-0
checkpoint-epoch-1
checkpoint-epoch-2
```

其中：

* `checkpoint-epoch-0`：完成第 1 个 epoch 后的模型；
* `checkpoint-epoch-1`：完成第 2 个 epoch 后的模型；
* `checkpoint-epoch-2`：完成第 3 个 epoch 后的模型。

### 15.1 Loss 记录

部分训练日志：

```text
Epoch 0 | Step 0  | Loss: 13.6357
Epoch 0 | Step 10 | Loss: 13.6513
Epoch 0 | Step 20 | Loss: 11.9782
Epoch 0 | Step 30 | Loss: 12.7955
Epoch 0 | Step 40 | Loss: 11.0931

Epoch 1 | Step 0  | Loss: 10.9201
Epoch 1 | Step 10 | Loss: 11.2084
Epoch 1 | Step 20 | Loss: 10.2244
Epoch 1 | Step 30 | Loss: 10.1893
Epoch 1 | Step 40 | Loss: 8.9955

Epoch 2 | Step 0  | Loss: 9.2504
Epoch 2 | Step 10 | Loss: 9.4519
Epoch 2 | Step 20 | Loss: 8.4935
Epoch 2 | Step 30 | Loss: 8.2761
Epoch 2 | Step 40 | Loss: 7.9727
```

从日志可以看出，loss 总体由约：

```text
13.64
```

下降到：

```text
7.97
```

说明模型在训练数据上进行了有效学习。

完整 loss 摘要位于：

```text
logs/exp01_losses.txt
```

需要注意的是，训练 loss 下降只说明模型对训练目标的拟合增强，并不能单独证明生成语音一定更自然或音色一定更相似。因此仍需通过生成音频进行主观对比。

---

## 16. 微调后推理

本实验对 3 个 checkpoint 使用相同的 3 类测试文本进行生成。

推理脚本：

```text
scripts/infer_exp01_compare.py
```

运行示例：

```bash
python scripts/infer_exp01_compare.py \
  --checkpoint-root /path/to/exp01_lr2e-6_bs2_ep3 \
  --output-dir /path/to/exp01_compare \
  --speaker-name student_voice
```

推理使用：

```text
dtype = torch.bfloat16
attention = flash_attention_2
language = Chinese
speaker = student_voice
```

---

## 17. 测试文本设计

每个 checkpoint 使用以下 3 类文本。

### 17.1 接近训练数据风格的文本

文件标记：

```text
seen_style
```

文本：

```text
如果时间允许，我们下午一起去实验室继续完成模型训练。
```

用途：

* 测试模型对训练数据相近表达风格的掌握情况；
* 观察微调后音色和语气是否稳定。

### 17.2 未见短句

文件标记：

```text
unseen_short
```

文本：

```text
这是使用我的个人语音数据微调后的测试语音。
```

用途：

* 测试模型对未参与训练短文本的泛化能力；
* 观察基本发音准确度和音色相似度。

### 17.3 未见长句

文件标记：

```text
unseen_long
```

文本：

```text
人工智能语音合成技术正在快速发展，并逐渐应用到教育、医疗和智能交互等实际场景。
```

用途：

* 测试模型的长文本生成稳定性；
* 观察断句、语速、重复、漏字和错读问题；
* 比较不同 epoch 的泛化能力。

---

## 18. 生成结果

实验采用：

```text
3 个 checkpoint × 3 类文本
```

因此共生成：

```text
9 条语音
```

文件包括：

```text
epoch0_seen_style.wav
epoch0_unseen_short.wav
epoch0_unseen_long.wav

epoch1_seen_style.wav
epoch1_unseen_short.wav
epoch1_unseen_long.wav

epoch2_seen_style.wav
epoch2_unseen_short.wav
epoch2_unseen_long.wav
```

生成结果位于：

```text
results/samples/
```

文件命名含义：

* `epoch0`：第 1 个 epoch checkpoint；
* `epoch1`：第 2 个 epoch checkpoint；
* `epoch2`：第 3 个 epoch checkpoint；
* `seen_style`：接近训练数据风格的文本；
* `unseen_short`：未见短句；
* `unseen_long`：未见长句。

---

## 19. 结果评价方法

本实验主要使用主观试听进行 checkpoint 对比。

评价维度包括：

### 19.1 音色相似度

判断生成声音是否接近目标说话人的：

* 声线；
* 音高范围；
* 共鸣特征；
* 发声习惯；
* 整体听感。

### 19.2 自然度

判断生成语音是否存在：

* 明显机器感；
* 发音生硬；
* 音高抖动；
* 不自然停顿；
* 语速异常；
* 句尾拖音；
* 韵律不连贯。

### 19.3 文本准确度

检查是否存在：

* 漏字；
* 多字；
* 错读；
* 重复；
* 发音模糊；
* 数字或专有名词错误。

### 19.4 长文本泛化

重点观察未见长句中的：

* 断句是否合理；
* 前后音色是否一致；
* 后半句是否退化；
* 是否出现重复或提前结束；
* 整句韵律是否自然。

训练 loss 最低的 checkpoint 不一定是主观效果最好的 checkpoint，因此需要综合试听结果选择最终模型。

---

## 20. 仓库目录结构

```text
qwen3_tts_finetuning/
├── README.md
├── requirements.txt
├── environment.txt
├── .gitignore
│
├── configs/
│   └── exp01.yaml
│
├── scripts/
│   ├── create_finetune_jsonl.py
│   ├── infer_exp01_compare.py
│   ├── sft_12hz_modified.py
│   └── sft_12hz_original.py
│
├── metadata/
│   ├── train_raw.example.jsonl
│   ├── train_ids.txt
│   └── test_ids.txt
│
├── logs/
│   ├── exp01_environment.txt
│   └── exp01_losses.txt
│
└── results/
    └── samples/
        ├── README.md
        ├── epoch0_seen_style.wav
        ├── epoch0_unseen_short.wav
        ├── epoch0_unseen_long.wav
        ├── epoch1_seen_style.wav
        ├── epoch1_unseen_short.wav
        ├── epoch1_unseen_long.wav
        ├── epoch2_seen_style.wav
        ├── epoch2_unseen_short.wav
        └── epoch2_unseen_long.wav
```

---

## 21. 文件说明

### `configs/exp01.yaml`

保存第一轮微调实验的：

* 模型名称；
* 数据规模；
* 硬件信息；
* batch size；
* 梯度累积；
* 学习率；
* epoch 数；
* loss 结果。

### `scripts/create_finetune_jsonl.py`

用于：

* 读取音频目录和文本文件；
* 检查音频文件；
* 按固定规则划分训练集和测试集；
* 创建官方微调所需的原始 JSONL。

### `scripts/infer_exp01_compare.py`

用于：

* 依次加载 3 个 checkpoint；
* 对固定的 3 条测试文本进行生成；
* 保存 9 条对比语音；
* 自动释放不同 checkpoint 之间的 GPU 显存。

### `scripts/sft_12hz_original.py`

保存实验所使用的原始训练脚本版本，便于与修改版对比。

### `scripts/sft_12hz_modified.py`

在原始脚本基础上增加或调整了 TensorBoard 日志目录配置，以兼容当前版本的 Accelerate。

### `metadata/train_ids.txt`

记录 90 条训练样本编号。

### `metadata/test_ids.txt`

记录 10 条测试样本编号。

### `metadata/train_raw.example.jsonl`

展示训练 JSONL 的字段格式，不包含真实个人数据。

### `logs/exp01_losses.txt`

记录训练过程中打印的 loss。

### `logs/exp01_environment.txt`

记录实验环境、GPU、PyTorch、CUDA、Transformers 和代码 commit 等信息。

### `results/samples/`

保存 9 条微调后生成的测试语音。

---

## 22. 未上传的内容

由于文件体积、模型许可和个人隐私原因，本仓库不上传以下内容：

```text
Qwen3-TTS 基础模型
Qwen3-TTS Tokenizer
微调 checkpoint
原始个人录音
统一参考音频
完整训练文本
train_raw.jsonl
train_with_codes.jsonl
Hugging Face 缓存
Conda 环境目录
TensorBoard 原始事件文件
```

其中微调 checkpoint 体积较大，也不适合使用普通 Git 进行版本管理。

若后续需要共享模型，可考虑：

* Hugging Face Hub 私有模型仓库；
* Git LFS；
* 云盘；
* 对象存储；
* 学校内部服务器。

---

## 23. 隐私说明

本实验使用个人语音数据。

仓库中公开的生成音频可能保留部分目标说话人的音色特征。上传生成结果前，应确认目标说话人同意公开这些音频。

本仓库未公开：

* 原始录音；
* 参考音频；
* 完整训练数据；
* 模型 checkpoint。

不得将本项目用于：

* 未经授权的声音冒用；
* 身份欺骗；
* 伪造他人语音；
* 侵犯隐私或人格权的用途。

---

## 24. 已知问题

### 24.1 FlashAttention 安装兼容性

FlashAttention 对以下版本组合较敏感：

```text
Python
PyTorch
PyTorch CUDA Runtime
系统 CUDA Toolkit
C++ ABI
FlashAttention 版本
```

若出现：

```text
undefined symbol
CUDA version mismatch
```

应检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```

确保 PyTorch CUDA Runtime 与本地 CUDA Toolkit 版本兼容。

### 24.2 Accelerate TensorBoard 配置

较新版本的 Accelerate 在：

```python
log_with="tensorboard"
```

时需要指定日志目录，例如：

```python
Accelerator(
    gradient_accumulation_steps=4,
    mixed_precision="bf16",
    log_with="tensorboard",
    project_dir="logs/tensorboard/exp01",
)
```

### 24.3 小数据过拟合

本实验仅有 90 条训练语音，全参数微调可能较快过拟合。

增加 epoch 时，应持续比较：

* 未见文本自然度；
* 错读率；
* 长句稳定性；
* 音色相似度。

不应只根据训练 loss 选择 checkpoint。

---

## 25. 后续可扩展实验

本次实验已经完成基础微调闭环，后续还可以扩展以下内容。

### 25.1 增加训练轮数

可增加 5 epoch 实验：

```text
exp02_lr2e-6_bs2_ep5
```

用于观察：

* 音色相似度是否继续提升；
* 是否出现明显过拟合；
* 第 3、4、5 个 epoch 的自然度变化。

### 25.2 学习率对比

可以比较：

```text
1e-6
2e-6
5e-6
```

观察不同学习率对：

* 收敛速度；
* 音色学习；
* 发音准确率；
* 稳定性；

的影响。

### 25.3 LoRA 对比实验

当前实验为全参数 SFT。

后续可以实现 LoRA 微调，并比较：

```text
显存占用
训练时间
可训练参数量
checkpoint 大小
音色相似度
自然度
文本准确率
```

### 25.4 客观评价

可以增加：

* CER：字符错误率；
* SIM：说话人相似度；
* MOS：主观自然度评分；
* UTMOS：自动语音质量预测；
* 音频时长与文本长度分析；
* 不同 checkpoint 的推理速度。

### 25.5 Zero-shot 与微调模型对比

可以使用完全相同的文本比较：

```text
Zero-shot voice cloning
Fine-tuned epoch 0
Fine-tuned epoch 1
Fine-tuned epoch 2
```

从而评价微调相对于 zero-shot 的实际提升。

---

## 26. 实验结论

本实验成功完成了 Qwen3-TTS 从官方推理复现到个人语音全参数微调的完整流程。

主要结果如下：

1. 官方 zero-shot 推理成功运行；
2. 个人参考音频 zero-shot voice cloning 成功；
3. 100 条个人语音数据通过完整性和格式检查；
4. 90 条训练数据全部成功提取 `audio_codes`；
5. 在单张 NVIDIA A800 80GB 上完成 1.7B Base 模型全参数 SFT；
6. 训练完成 3 个 epoch，没有发生 OOM、NaN 或异常退出；
7. 训练 loss 从约 13.64 下降至约 7.97；
8. 成功保存 3 个 checkpoint；
9. 使用 3 个 checkpoint 生成共 9 条测试语音；
10. 实验代码、配置、环境、日志和部分生成结果已整理并上传至 GitHub。

本实验说明，在数量有限的个人中文语音数据上，Qwen3-TTS 官方单说话人微调流程可以成功运行，并使模型学习目标说话人的语音特征。

由于训练数据规模较小，最终 checkpoint 的选择不能只依赖训练 loss，还需结合音色相似度、自然度、文本准确度和未见文本泛化能力进行综合判断。

---

## 27. 最终完成状态

```text
官方推理复现：已完成
个人 zero-shot 推理：已完成
数据完整性检查：已完成
训练测试集划分：已完成
audio_codes 提取：已完成
FlashAttention 环境配置：已完成
全参数 SFT：已完成
3 个 checkpoint 保存：已完成
微调后语音生成：已完成
9 条结果音频保存：已完成
代码与实验记录上传 GitHub：已完成
```

本项目的核心实验流程已经全部完成。

---

需要注意一个表述：你的**核心实验任务已经完成**，但若课程要求必须提供 CER、SIM、MOS、训练曲线图或 zero-shot 与 fine-tuning 的定量对比，那么这些属于后续评估工作，需要根据老师的评分要求补充。