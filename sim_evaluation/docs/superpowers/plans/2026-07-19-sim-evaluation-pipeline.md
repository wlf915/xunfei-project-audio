# 自动化 SIM 评测流水线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建一个独立、可复现的中文说话人相似度评测模块，自动比较 zero-shot 与 SFT 合成语音。

**Architecture:** `sim_pipeline.py` 只负责清单发现、配对、聚合和 CSV 输出，接受可注入的评分函数，便于无模型单元测试。`run_sim.py` 负责 CLI 参数、FunASR ERes2NetV2 embedding 提取和余弦相似度计算，输出表格与图像。

**Tech Stack:** Python 3.12、FunASR ERes2NetV2、matplotlib、pytest。

## Global Constraints

- 评测数据必须与 SFT 训练数据分离。
- zero-shot 与 SFT 必须对同一批文件名的测试文本生成音频。
- 不修改 `基线代码/Qwen3-TTS-main/Qwen3-TTS-main` 中的模型源码。
- 真实评测采用 `iic/speech_eres2netv2_sv_zh-cn_16k-common`。

---

### Task 1: 建立可测试的数据发现与统计核心

**Files:**
- Create: `evaluation/tests/test_sim_pipeline.py`
- Create: `evaluation/sim_pipeline.py`
- Create: `evaluation/__init__.py`

**Interfaces:**
- Produces `discover_generated_audio(generated_root: Path) -> list[GeneratedSample]`。
- Produces `score_samples(samples, enrollment_files, scorer) -> list[PairScore]`。
- Produces `summarize_samples(pair_scores) -> tuple[pandas.DataFrame, pandas.DataFrame]`。

- [ ] **Step 1: Write failing tests** for matching system sample discovery, pair scoring, and system summary.
- [ ] **Step 2: Run `python -m pytest evaluation/tests/test_sim_pipeline.py -v`** and verify the tests fail because `sim_pipeline` does not exist.
- [ ] **Step 3: Implement only the discovered interfaces** with dataclasses, WAV validation, deterministic sorting, and injected scorer.
- [ ] **Step 4: Run the same test command** and verify all tests pass.

### Task 2: 实现可运行的 ERes2NetV2 CLI 与产物写入

**Files:**
- Create: `evaluation/run_sim.py`
- Modify: `evaluation/sim_pipeline.py`
- Test: `evaluation/tests/test_sim_pipeline.py`

**Interfaces:**
- `build_eres2net_scorer(model_id: str, device: str) -> Callable[[Path, Path], float]`。
- CLI accepts `--enrollment-dir`、`--generated-dir`、`--output-dir`、`--device`、`--model-id`.

- [ ] **Step 1: Write failing tests** for CSV output columns and mismatched zero-shot/SFT sample names.
- [ ] **Step 2: Run the targeted tests** and verify they fail for the missing writer/validation behavior.
- [ ] **Step 3: Implement CSV writers, matching-name validation, and CLI argument parsing.**
- [ ] **Step 4: Run all evaluation tests** and verify they pass without downloading ERes2NetV2.

### Task 3: 增加示例数据布局、依赖和中文说明

**Files:**
- Create: `evaluation/README.md`
- Create: `evaluation/requirements.txt`
- Create: `evaluation/data/enrollment/.gitkeep`
- Create: `evaluation/data/generated/zero_shot/.gitkeep`
- Create: `evaluation/data/generated/sft/.gitkeep`
- Create: `evaluation/examples/README.md`

- [ ] **Step 1: Document directory roles, data-leakage rules, installation, one-command execution, and output interpretation.**
- [ ] **Step 2: Include a dry-run command and a real ERes2NetV2 command.**
- [ ] **Step 3: Verify README paths and command names against the implemented CLI.**

### Task 4: 完整验证

**Files:**
- Test: `evaluation/tests/test_sim_pipeline.py`

- [ ] **Step 1: Run `python -m pytest evaluation/tests -v`.**
- [ ] **Step 2: Run `python evaluation/run_sim.py --help` and verify the required arguments are listed.**
- [ ] **Step 3: Report that real scoring additionally requires installation of `evaluation/requirements.txt` and first-run ERes2NetV2 model download.**
