# 记录你自己的验证答案复核

`summarize_recovery.py`只提出数值选择，不能自动判断答案的事实正确性。发布前请实际阅读验证答案，再用`record_quality_review.py`记录你自己写下的观察；不需要手工拼内部JSON，也不需要冒充项目当时的助手复核。

这个命令不是自动审核器。它只确认笔记非空、数值汇总完成、被选验证报告的SHA仍一致，并把笔记与报告绑定。`completed_operator_review`表示操作员通过此命令自声明完成了复核记录，不证明任何人确实读过答案，更不表示模型获得独立人工验收。

## 实际查看哪些内容

先完成两边0／16／32／64步的八份validation与`recovery_summary.json`。阅读汇总中的完整八行和同step对照，再打开它记录的被选检查点`report_path`及对应`qa_predictions.jsonl`。原始记录包含资料、问题、参考、模型原文和停止原因。与同分支step0及同预算另一分支对照，保留具体题目ID。

笔记应写你真实查看了哪些报告与题目，以及观察到了什么，例如连续重复、复述问题、答非所问、事实错误、遗漏或输出被长度上限截断。达到生成上限只是一个需要查看的信号，不等于重复；EM/F1变化也不能自动说明事实对错。

不要把示例文字当作已经完成的观察直接提交。这里只列可用的笔记结构，请在阅读之后用自己的事实填写：

```text
审阅范围：实际阅读的两个被选报告、step0对照，以及查看的题目ID。
观察到的问题：具体ID、答案现象、资料或参考所支持的判断。
未发现的问题：仅限实际看过的范围，说明检查方法，不扩大结论。
限制：未阅读的范围、未解决的错误、尚需进一步人工判断的情况。
当前判断：是否准备继续固定发布评测；若不准备，说明原因并停止该流程。
```

只对validation进行这一轮检查点选择与复核，不使用final回头改训练或选择。若笔记发现了不接受的问题，应停止发布流程；记录脚本不会替你作“通过”的判断。

## 绑定笔记，生成复核JSON

把实际观察保存为UTF-8文件，例如`my-review-notes.md`。在项目目录打开PowerShell，将下面路径改为自己本轮的新汇总和笔记文件：

```powershell
$taskPython = '.\.work\qwen-env\Scripts\python.exe'
& $taskPython -X utf8 .\scripts\record_quality_review.py --summary '.\results\my-recovery\summary\recovery_summary.json' --notes '.\my-review-notes.md' --output '.\results\my-recovery\operator_review.json'
if ($LASTEXITCODE -ne 0) { throw 'Review record was not created.' }
```

可以加`--reviewer`给自己填写一个标签；身份不会被验证。脚本不会生成观察内容，不会读最终题库，不会改选checkpoint，不会修改权重。已有输出文件一律拒绝覆盖。笔记原文、笔记SHA、汇总SHA、被选验证报告SHA与当前时间都保存在新JSON中。

该记录明确写入：`operator_review_self_declared=true`、`actual_reading_verified=false`、`factual_correctness_automatically_verified=false`、`independent_review_performed=false`、`human_acceptance_performed=false`。因此不能把它描述为“程序认证已完成人工验收”。项目历史的`completed_assistant_review`记录继续原样保留。

## 接入固定发布流程

继续`freeze_release_policy.py`时，把新生成的`operator_review.json`交给`--review`，把同一份`recovery_summary.json`交给`--selection`。冻结脚本同时支持历史助手记录与新的操作员自声明记录，且校验它们绑定的汇总SHA。实际是否继续发布由你阅读后的判断决定，不由记录脚本推断。

当前发布配方支持原模型与校准5888两边**均选中step64**的情况，并要求导出确实来自对应被选报告。如果复现在其他机器上选择了step16／32或step0，不要手改汇总让它看起来选中64，也不要放宽门槛；保留结果并停止这套固定step64发布配方。现有范围的复现不等于任意checkpoint的通用发布工具。
