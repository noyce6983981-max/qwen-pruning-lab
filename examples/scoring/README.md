# 第七课：先知道分数在测什么

本目录的题目和回答全部是手写教学数据，**不是任何模型的实测输出**。我们故意写错日期、加上否定，并漏掉一道题，观察评分器的行为。

- `questions.jsonl`：每行一题，包含唯一 `id`、`question` 和非空 `answers` 列表。可添加 `document` 等字段，评分器只读取上述三个字段。
- `predictions_handwritten.jsonl`：每行一个回答，包含对应 `id` 和 `prediction`。未知或重复 ID 会报错；漏答按 0 分计入总题数。

在项目目录运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 score`，无需加载模型或显卡。

预期看到：格式题会忽略全角、大小写和标点差异，完全匹配；错误日期和“不需要”的完全匹配为 0，但字符 F1 仍较高；第四题缺失、记 0 分。字符 F1 衡量字面重合，不能判断一句话是否说反了。

亲手练习：把 `deadline` 的回答改成 `周五下午五点`，再次运行，看逐题分数和均值的变化。

这个评分器也接受你自己的同格式 JSONL 文件。直接调用 `scripts/score_qa.py --questions <题库路径> --predictions <回答路径> --output-dir <新的结果目录>`，即可生成 JSON 与 Markdown 报告。Python 使用项目环境或任意支持当前脚本语法的 Python 3.10+；模块仅依赖标准库。

对比模型时，两边必须使用同一题库、参考答案和评分规则。此处是教学用字符 F1，正式采用某个公开数据集时还需使用该数据集的官方评分口径。自定义题库的模型批量推理入口尚待接入。
