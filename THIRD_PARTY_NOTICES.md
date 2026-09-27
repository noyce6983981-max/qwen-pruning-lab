# 来源与许可

本项目自有代码及说明采用 MIT，见 [LICENSE](LICENSE)。以下材料不因根目录许可证而改变许可。

| 材料 | 来源与范围 | 许可与处理 |
| --- | --- | --- |
| Qwen3-1.7B | [官方模型](https://huggingface.co/Qwen/Qwen3-1.7B)，revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0；仓库不携带模型权重，下载源模型时附带其许可。以后分发改动后的权重须单独整理源许可、归属与改动说明。 |
| CMRC2018 | Cui 等人，[论文](https://aclanthology.org/D19-1600/)，[固定源提交](https://github.com/ymcui/cmrc2018/tree/c0eb1b6ba219847457e6af3180da722bbeb656af) | CC BY-SA 4.0；[许可全文](third_party/cmrc2018/LICENCE)。官方评分器原字节保留，数据运行时下载。 |
| CMRC 派生材料 | 本地子集按资料分组、长度筛选并转换为 JSONL；报告中的题目、答案、引用和派生结果 | 此类派生部分按 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 分享，保留本说明。不是官方隐藏测试成绩。 |
| CMRC Python 3 适配 | 本项目 `scripts/cmrc_official.py` | 仅在内存中做有限的兼容替换；原公式和来源见 [评分器说明](third_party/cmrc2018/README.md)。不声称官方发布了本适配。 |
| NLTK 3.9.2 | [NLTK](https://github.com/nltk/nltk) | 代码 Apache-2.0；保留 [许可文本](third_party/cmrc2018/NLTK_LICENSE.txt)。 |
| English Punkt 数据 | [NLTK 数据固定提交](https://github.com/nltk/nltk_data/tree/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a) | 上游清单中的模型许可未明确，仓库不再分发缓存；提供固定下载器、来源和 SHA256。NLTK 代码许可不覆盖该模型许可问题。 |

校准评分参考 [FLAP 固定版本的 Wanda-style / WIFN 形式](https://github.com/CASIA-LMC-Lab/FLAP/blob/3bb57db3449dd2fa04a5c2192de80e87e33be2b1/lib/prune.py)。实现只做 Qwen3 MLP 适配，使用未剪模型收集激活，未实现 FLAP 的完整逐层传播、方差评分或补偿。有限 LoRA 恢复参考 [LLM-Pruner](https://github.com/horseee/LLM-Pruner) 的流程思路，使用 PEFT 实现；不称完整论文复现。

依赖包仍遵循各自许可证。仓库列出安装依赖，不将虚拟环境、字体文件或依赖二进制作为项目源码发布。
