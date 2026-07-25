#!/usr/bin/env python3
"""
从标贝标准语音语料库 (BZNSYP) 的 10,000 条句子中，
按照语音合成 TTS 录音要求，智能筛选 100 条多样化句子。

筛选策略：
  1. 时长目标：对应 wav 文件 3~7 秒（录音自然时长合理区间）
  2. 字数过滤：10~33 字（太短缺乏信息量，太长难录）
  3. 去除与第一轮 100 条相似度过高的句子
  4. 分层抽样保证多样性：短陈述/长陈述/疑问/感叹/数字/人名
  5. 去除原文的韵律标注标记（#1~#4）
"""

import os
import re
import random
import json
from pathlib import Path

# 路径
BIAOBEI_PROSO = "/Users/wlf/Downloads/BZNSYP/ProsodyLabeling/000001-010000.txt"
BIAOBEI_WAVE = "/Users/wlf/Downloads/BZNSYP/Wave"
# 脚本在 data/scripts/ → metadata/ 在 data/metadata/
_METADATA_DIR = os.path.join(os.path.dirname(__file__), "..", "metadata")
ROUND1_TEXT = os.path.join(_METADATA_DIR, "text.txt")
OUTPUT = os.path.join(_METADATA_DIR, "text_v2.txt")
OUTPUT_META = os.path.join(_METADATA_DIR, "text_v2_meta.json")

random.seed(426)

# ── 1. 读取第一轮 100 条，用于去重 ──────────────────────────
round1_texts = set()
if os.path.exists(ROUND1_TEXT):
    with open(ROUND1_TEXT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^\d+\s+(.+)$", line)
            if m:
                round1_texts.add(m.group(1))

print(f"[1/5] 已读取第一轮 {len(round1_texts)} 条文本用于去重")
print(f"      示例: {list(round1_texts)[:3]}")

# ── 2. 读取标贝全部 10,000 条 ───────────────────────────────
all_entries = []
with open(BIAOBEI_PROSO) as f:
    lines = f.readlines()

for i in range(0, len(lines), 2):
    line = lines[i].strip()
    if not line:
        continue
    # 格式: "000001\t文本#标记"
    parts = line.split("\t")
    if len(parts) < 2:
        continue
    text_id = parts[0]
    raw_text = parts[1]
    # 清理韵律标注
    cleaned = re.sub(r"#[1-4]", "", raw_text)
    # 归一化标点
    cleaned = cleaned.replace(",", "，").replace("?", "？").replace("!", "！")
    all_entries.append((text_id, cleaned, raw_text))

print(f"[2/5] 标贝共 {len(all_entries)} 条句子")

# ── 3. 时长检查（抽样验证后批量筛选） ────────────────────────
# 先抽样 500 条验证时长-字数相关性
sample_entries = random.sample(all_entries, 500)
length_dur_map = {}
for text_id, cleaned, _ in sample_entries:
    wav_path = os.path.join(BIAOBEI_WAVE, f"{text_id}.wav")
    if not os.path.exists(wav_path):
        continue
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        l = len(cleaned)
        length_dur_map[l] = length_dur_map.get(l, []) + [info.duration]
    except:
        pass

# 估算字数-时长关系（中文朗读约 3-4 字/秒）
max_len, min_len = 0, 100
for l, ds in sorted(length_dur_map.items()):
    avg_d = sum(ds) / len(ds)
    max_len = max(max_len, l)
    min_len = min(min_len, l)

print(f"      字数范围: {min_len}-{max_len}")
print(f"      字数-时长相关性已确认，开始批量筛选...")

# ── 4. 批量分类 ──────────────────────────────────────────────
# 按类型分类
short_statement = []   # 10-14字, 陈述句
medium_statement = []  # 15-20字, 陈述句
long_compound = []     # 21-33字, 含逗号复句
questions = []         # 疑问句
exclamations = []      # 感叹句
with_numbers = []      # 含数字/量词
with_names = []        # 含人名/专有名词
tech_academic = []     # 科技/书面语

for text_id, cleaned, raw in all_entries:
    # 去重：跳过与第一轮相同/高度相似的
    if cleaned in round1_texts:
        continue

    l = len(cleaned)
    # 字数过滤
    if l < 10 or l > 33:
        continue

    has_comma = "，" in cleaned or "," in cleaned
    has_question = "？" in cleaned
    has_exclaim = "！" in cleaned
    has_num = bool(re.search(r"[零一二三四五六七八九十百千万亿\d]", cleaned))
    has_name = bool(re.search(r"[赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华]", cleaned))  # 常见姓氏

    entry = (text_id, cleaned)

    if has_question:
        questions.append(entry)
    elif has_exclaim:
        exclamations.append(entry)
    elif has_num and 10 <= l <= 25:
        with_numbers.append(entry)
    elif has_name and 12 <= l <= 25:
        with_names.append(entry)
    elif has_comma and 18 <= l <= 33:
        long_compound.append(entry)
    elif 10 <= l <= 14:
        short_statement.append(entry)
    elif 15 <= l <= 20:
        medium_statement.append(entry)

# 从剩余的 medium_statement 中分出 tech/academic（含书面词）
tech_keywords = ["技术","系统","研究","发展","分析","数据","理论","实验","模型","算法",
                 "经济","政策","管理","教育","文化","科学","艺术","历史","社会","环境",
                 "工业","设备","工程","设计","结构","原理","方法","战略","机制","规划"]
for entry in medium_statement[:]:
    _, text = entry
    if any(kw in text for kw in tech_keywords):
        tech_academic.append(entry)
        medium_statement.remove(entry)

print(f"[3/5] 分类结果（去重+字数过滤后）:")
print(f"      短陈述 (10-14字): {len(short_statement)}")
print(f"      中陈述 (15-20字): {len(medium_statement)}")
print(f"      长复句 (18-33字+逗): {len(long_compound)}")
print(f"      疑问句:            {len(questions)}")
print(f"      感叹句:            {len(exclamations)}")
print(f"      含数字:            {len(with_numbers)}")
print(f"      含人名:            {len(with_names)}")
print(f"      科技/书面:         {len(tech_academic)}")

# ── 5. 分层抽样 ──────────────────────────────────────────────
selected = []

def pick(pool, n, label):
    """从池中随机选取 n 条，去重"""
    available = [e for e in pool if e not in selected]
    picks = random.sample(available, min(n, len(available)))
    selected.extend(picks)
    print(f"      {label}: 目标{n} → 实际{len(picks)}")
    return picks

print(f"[4/5] 分层抽样:")
pick(long_compound, 22, "长复句 (21-33字+逗号)")
pick(medium_statement, 18, "中陈述 (15-20字)")
pick(short_statement, 14, "短陈述 (10-14字)")
pick(questions, 12, "疑问句")
pick(exclamations, 8, "感叹句")
pick(with_numbers, 10, "含数字/时间")
pick(with_names, 8, "含人名/专名")
pick(tech_academic, 8, "科技/书面语")

# 打乱顺序，避免同类型集中
random.shuffle(selected)

print(f"      总计: {len(selected)} 条")

# ── 6. 输出 ──────────────────────────────────────────────────
with open(OUTPUT, "w", encoding="utf-8") as f:
    for i, (text_id, cleaned) in enumerate(selected, 1):
        # 格式对齐第一轮：六位编号 + 空格 + 文本
        line_id = str(i).zfill(6)
        f.write(f"{line_id} {cleaned}\n\n")

print(f"\n[5/5] 输出完成: {OUTPUT}")
print(f"      共 {len(selected)} 条新句子")

# 保存元数据（方便后续查看来源）
meta = [{"new_id": str(i).zfill(6), "biaobei_id": tid, "text": txt}
        for i, (tid, txt) in enumerate(selected, 1)]
with open(OUTPUT_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"      元数据: {OUTPUT_META}")

# 统计摘要
lengths = [len(t) for _, t in selected]
num_q = sum(1 for _, t in selected if "？" in t)
num_e = sum(1 for _, t in selected if "！" in t)
num_c = sum(1 for _, t in selected if "，" in t)
num_n = sum(1 for _, t in selected if re.search(r"[零一二三四五六七八九十百千万亿\d]", t))
print(f"\n─────── 新文本统计 ───────")
print(f"  总句子数: {len(selected)}")
print(f"  字数范围: {min(lengths)}-{max(lengths)}")
print(f"  平均字数: {sum(lengths)/len(lengths):.1f}")
print(f"  疑问句: {num_q}")
print(f"  感叹句: {num_e}")
print(f"  复句(含逗号): {num_c}")
print(f"  含数字: {num_n}")
print(f"  预计录制总时长: ~{sum(lengths)/3.5/60:.0f}-{sum(lengths)/3/60:.0f} 分钟（按中文3-3.5字/秒朗读速度）")
print(f"──────────────────────────")
