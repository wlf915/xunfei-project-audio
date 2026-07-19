# 文件命名示例

同一条测试文本在两个系统中的音频必须同名：

```text
data/generated/zero_shot/test_001.wav
data/generated/sft/test_001.wav
```

例如 `test_001.wav` 可以都表示文本“轻轻的我走了，正如我轻轻地来”。

不要把音频本体提交到代码工程；真实音频只需放入 `evaluation/data/` 对应目录，或在运行命令中传入你的绝对路径。
