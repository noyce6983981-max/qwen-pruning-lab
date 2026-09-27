# CMRC2018 官方评分来源

`cmrc2018_evaluate.py` 和 `LICENCE` 从 [CMRC2018 官方仓库](https://github.com/ymcui/cmrc2018/tree/c0eb1b6ba219847457e6af3180da722bbeb656af) 原样保存，未修改字节。固定提交为 `c0eb1b6ba219847457e6af3180da722bbeb656af`，评分脚本为 `squad-style-data/cmrc2018_evaluate.py` 的 v5-special 版本。SHA256 和下载地址见 `provenance.json`。仓库许可为 CC-BY-SA-4.0，原许可全文保留在 `LICENCE`。

上游脚本使用 Python 2。本项目 `scripts/cmrc_official.py` 在内存中做四类经过出现次数检查的 Python 3 兼容替换：删除 Python 2 默认编码设置、去掉两处输入字符串的 UTF-8 decode、把 `ur` 正则前缀换成 `r`、去掉预测字符串的 UTF-8 decode。输入严格限制为 Unicode 字符串。原始文件保持不变；分词调用、标点表、最长连续公共片段计算和多参考答案取最高分的公式均未改动。

注意官方 F1 使用**最长连续公共 token 片段**，不是字符多重集重合，也不是非连续的最长公共子序列。中文按单字切分，英文部分调用 `nltk.word_tokenize`。官方 EM 只删除源码列出的标点，保留内部空白，不做 NFKC。不要额外清洗输入后仍声称它与原始输出的官方指标相同。

适配层调用官方 `evaluate()` 汇总，并调用同一脚本的两个单题函数输出逐题 EM/F1，所有值均为百分数。它额外拒绝重复样本 ID、漏预测、多余预测、无效参考答案和重复 JSON 键，避免悄悄改变分母。17 项 CPU 单测检查官方汇总／单题一致性，以及中文、英文、数字、空输出、标点、重复字符等边界；这不等同于复现 2018 年未锁定的完整 Python 2/NLTK 运行环境，也不是官方隐藏测试成绩。

## NLTK 及资源

当前项目明确固定 `nltk==3.9.2`。NLTK 代码采用 Apache-2.0，许可原文在 `NLTK_LICENSE.txt`。安装和准备命令（在项目根目录运行，`python` 需指向本项目环境）：

```powershell
python -m pip install -r requirements-windows.txt
python third_party/cmrc2018/fetch_punkt.py
python -m unittest discover -s tests -p test_cmrc_official.py -v
```

按公开教程安装后，可将 `python` 替换为 `./.work/qwen-env/Scripts/python.exe`。首次需要显式准备资源；以后下载器会校验已有缓存。评分时不会联网或自动下载。

English Punkt 来自 [NLTK 官方数据仓库固定提交](https://github.com/nltk/nltk_data/tree/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a) 的 `packages/tokenizers/punkt_tab.zip`。下载器校验完整归档的 SHA256，只提取英文所需的四个文件及原始 README。四个模型文件合计 237,757 字节，首次归档下载约 4.3 MB。提交号、来源 URL、归档及每个提取文件的哈希见 `nltk_punkt_provenance.json`。

**Punkt 模型许可未明确**：[NLTK 官方许可清单](https://github.com/nltk/nltk_data/blob/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/DATASET-LICENSES.md) 将 `punkt_tab` 列入许可不明确一组。NLTK 代码的 Apache 许可不能自动视作这些模型的许可。本项目把下载的模型放在被 `.gitignore` 排除的 `nltk_data/` 本地缓存中，不将其作为本项目代码再分发。发布时保留下载器和来源／哈希记录；不要将该缓存打包入代码发布物。

原始指标由 Cui 等人的 CMRC2018 数据集配套提供；使用数据开展研究时引用其 [EMNLP-IJCNLP 2019 论文](https://aclanthology.org/D19-1600/)。
