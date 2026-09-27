"""CPU-only exercise: change PREDICTION and observe official CMRC EM/F1."""
from cmrc_official import score_examples

# 只改这一行，再运行。先试完整答案，再试“北京”，最后试“上海大学”。
PREDICTION = "北京大学"


def main():
    example = {"id": "lesson_only", "answers": ["北京大学"]}
    scores = score_examples([example], {"lesson_only": PREDICTION})
    print("教学例子，不属于验证集或最终评测集。")
    print(f"标准答案：北京大学；模型答案：{PREDICTION}")
    print(f"EM = {scores['em']:.2f}；F1 = {scores['f1']:.2f}")
    print("EM 检查官方归一化后是否完全匹配；F1 给部分文字重合一些分数。")
    print("注意：上海大学和北京大学是不同学校；部分重合得分不等于事实正确。")


if __name__ == "__main__":
    main()
