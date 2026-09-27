"""CPU audits for leakage prevention and the frozen quality_v1 data protocol.

Synthetic grouping tests need no download or tokenizer. Frozen-data tests skip
explicitly if artifacts are absent; tokenizer checks also skip if the original
local tokenizer is unavailable. No language model or generated answer is used.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_quality_data import assign_groups, normalized


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_document(identifier, title, contexts, source="train"):
    return {"id": identifier, "title": title, "source": source,
            "paragraphs": [{"id": f"{identifier}-{i}", "context": text, "qas": []} for i, text in enumerate(contexts)]}


class GroupingTests(unittest.TestCase):
    def test_matching_normalized_title_groups_different_paragraphs(self):
        documents = [make_document("a", "ＡＢＣ\u3000百科", ["甲资料"]),
                     make_document("b", "abc 百科", ["乙资料"]),
                     make_document("c", "不同标题", ["丙资料"])]
        assign_groups(documents)
        self.assertEqual(documents[0]["group_id"], documents[1]["group_id"])
        self.assertNotEqual(documents[0]["group_id"], documents[2]["group_id"])

    def test_shared_context_connects_train_and_dev_despite_different_titles(self):
        documents = [make_document("a", "甲", ["Ｃａｆé\n资料"]),
                     make_document("b", "乙", ["café 资料"], source="dev")]
        assign_groups(documents)
        self.assertEqual(documents[0]["group_id"], documents[1]["group_id"])

    def test_transitive_title_and_context_bridge_protects_entire_train_group(self):
        # A shares a title with B; B shares one paragraph with C in dev.
        # Reserving dev C must also reserve A's otherwise unrelated paragraph.
        documents = [make_document("a", "共同标题", ["独立甲段落"]),
                     make_document("b", "共同标题", ["桥接段落", "独立乙段落"]),
                     make_document("c", "开发集标题", ["桥接段落"], source="dev"),
                     make_document("d", "无关标题", ["无关段落"])]
        assign_groups(documents)
        self.assertEqual(len({d["group_id"] for d in documents[:3]}), 1)
        self.assertNotEqual(documents[0]["group_id"], documents[3]["group_id"])
        dev_groups = {d["group_id"] for d in documents if d["source"] == "dev"}
        eligible_train = [d["id"] for d in documents if d["source"] == "train" and d["group_id"] not in dev_groups]
        self.assertEqual(eligible_train, ["d"])

    def test_group_ids_are_invariant_to_document_and_paragraph_order(self):
        original = [make_document("a", "共享", ["独立", "桥接"]),
                    make_document("b", "另一标题", ["桥接"], source="dev"),
                    make_document("c", "共享", ["第三段"])]
        expected = None
        for permutation in itertools.permutations(original):
            documents = copy.deepcopy(list(permutation))
            for document in documents:
                document["paragraphs"].reverse()
            assign_groups(documents)
            actual = {d["source"] + ":" + d["id"]: d["group_id"] for d in documents}
            if expected is None:
                expected = actual
            self.assertEqual(actual, expected)

    def test_blank_titles_do_not_merge_unrelated_contexts_or_claim_semantic_dedup(self):
        documents = [make_document("a", "", ["小明喜欢苹果。"]),
                     make_document("b", " \n", ["小明喜欢苹果！"])]
        assign_groups(documents)
        self.assertNotEqual(documents[0]["group_id"], documents[1]["group_id"])
        self.assertNotEqual(normalized(documents[0]["paragraphs"][0]["context"]),
                            normalized(documents[1]["paragraphs"][0]["context"]))


class FrozenDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / "data" / "quality_v1"
        manifest_file = cls.directory / "manifest.json"
        if not manifest_file.exists():
            raise unittest.SkipTest("quality_v1 frozen artifacts are not present")
        cls.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        cls.splits = {name: read_jsonl(cls.directory / f"{name}.jsonl") for name in cls.manifest["counts"]}
        cls.groups = json.loads((cls.directory / "groups.json").read_text(encoding="utf-8"))

    @classmethod
    def raw_documents(cls):
        result = []
        for source in ("train", "dev"):
            path = ROOT / "data" / "raw" / "cmrc2018" / f"{source}.json"
            if not path.exists():
                raise unittest.SkipTest("raw pinned CMRC source files are unavailable")
            result.extend({**document, "source": source} for document in json.loads(path.read_text(encoding="utf-8"))["data"])
        return result

    def test_frozen_files_and_available_sources_match_manifest_hashes(self):
        for filename, expected in self.manifest["files_sha256"].items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((self.directory / filename).read_bytes()).hexdigest(), expected)
        for source, metadata in self.manifest["source"].items():
            self.assertIn(self.manifest["revision"], metadata["url"])
            path = ROOT / "data" / "raw" / "cmrc2018" / f"{source}.json"
            if path.exists():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["sha256"])
        self.assertEqual(self.manifest["license"], "CC-BY-SA-4.0")
        self.assertIn(self.manifest["revision"], self.manifest["license_url"])
        self.assertIn("Cui", self.manifest["citation"])

    def test_every_purpose_is_group_disjoint_and_matches_locked_assignments(self):
        owners = {}
        context_owners = {}
        for name, rows in self.splits.items():
            self.assertEqual(len(rows), self.manifest["counts"][name]["rows"])
            self.assertEqual(len({row["group_id"] for row in rows}), self.manifest["counts"][name]["groups"])
            self.assertEqual(len({row["id"] for row in rows}), len(rows))
            for row in rows:
                self.assertEqual(row["source"], "dev" if name.startswith("final_") else "train")
                self.assertEqual(self.groups["assignments"][row["group_id"]], name)
                self.assertEqual(owners.setdefault(row["group_id"], name), name)
                self.assertEqual(context_owners.setdefault(row["normalized_context_sha256"], name), name)
        self.assertEqual(owners, self.groups["assignments"])

    def test_real_group_ids_recompute_and_no_selected_train_group_overlaps_dev(self):
        documents = self.raw_documents()
        assign_groups(documents)
        actual = {(d["source"], d["id"]): d["group_id"] for d in documents}
        recorded = {(d["source"], d["id"]): d["group_id"] for d in self.groups["documents"]}
        self.assertEqual(actual, recorded)
        dev_groups = {d["group_id"] for d in documents if d["source"] == "dev"}
        train_groups = {r["group_id"] for rows in self.splits.values() for r in rows if r["source"] == "train"}
        self.assertFalse(dev_groups & train_groups)

    def test_exact_qa_calibration_recovery_and_lm_budgets(self):
        for name, count in {"smoke_qa": 10, "validation_qa": 100, "final_qa": 200, "recovery": 512, "calibration": 64}.items():
            self.assertEqual(len(self.splits[name]), count)
        self.assertEqual({len(row["token_ids"]) for row in self.splits["calibration"]}, {256})
        self.assertEqual(sum(len(row["token_ids"]) for row in self.splits["calibration"]), 16384)
        for name, budget in {"smoke_lm": 1024, "validation_lm": 8192, "final_lm": 16384}.items():
            rows = self.splits[name]
            self.assertTrue(all(len(row["token_ids"]) >= 2 for row in rows))
            self.assertEqual(sum(len(row["token_ids"]) - 1 for row in rows), budget)
            self.assertEqual(self.manifest["counts"][name]["valid_lm_targets"], budget)
            for row in rows[:-1]:
                self.assertEqual(len(row["token_ids"]), row["original_token_count"])
            self.assertLessEqual(len(rows[-1]["token_ids"]), rows[-1]["original_token_count"])
        self.assertTrue(all(row["recovery_tokens_first_answer"] <= 512 for row in self.splits["recovery"]))

    def test_qa_text_and_all_references_are_untruncated_source_values(self):
        raw = {}
        for document in self.raw_documents():
            for paragraph in document["paragraphs"]:
                for question in paragraph["qas"]:
                    raw[(document["source"], question["id"])] = (paragraph["context"], question)
        for name, rows in self.splits.items():
            if not (name.endswith("_qa") or name == "recovery"):
                continue
            for row in rows:
                context, question = raw[(row["source"], row["id"])]
                self.assertEqual(row["context"], context)
                self.assertEqual(row["question"], question["question"])
                self.assertEqual(row["answers"], list(dict.fromkeys(a["text"] for a in question["answers"])))
                self.assertTrue(all(answer and answer in context for answer in row["answers"]))
                self.assertEqual(row["context_sha256"], hashlib.sha256(context.encode()).hexdigest())
                self.assertEqual(row["normalized_context_sha256"], hashlib.sha256(normalized(context).encode()).hexdigest())

    def test_frozen_tokens_match_original_tokenizer_prompts_and_text_prefixes(self):
        directory = Path(self.manifest["tokenizer"]["directory"])
        if not directory.is_dir():
            self.skipTest("original local tokenizer is unavailable")
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers is unavailable; token audit requires the project environment")
        for filename, expected in self.manifest["tokenizer"]["files_sha256"].items():
            self.assertEqual(hashlib.sha256((directory / filename).read_bytes()).hexdigest(), expected)
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        raw_contexts = {(d["source"], p["id"]): p["context"] for d in self.raw_documents() for p in d["paragraphs"]}
        protocol = self.manifest["protocol"]["qa"]
        for name, rows in self.splits.items():
            for row in rows:
                if name.endswith("_qa") or name == "recovery":
                    prompt = tokenizer.apply_chat_template([
                        {"role": "system", "content": protocol["system_prompt"]},
                        {"role": "user", "content": protocol["user_template"].format(context=row["context"], question=row["question"])},
                    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    tokens = tokenizer.encode(prompt, add_special_tokens=False)
                    self.assertEqual(tokens, row["prompt_token_ids"])
                    self.assertEqual(len(tokens), row["input_tokens"])
                    self.assertLessEqual(len(tokens), 448)
                    answer_lengths = [len(tokenizer.encode(answer, add_special_tokens=False)) for answer in row["answers"]]
                    self.assertTrue(all(length <= 63 for length in answer_lengths))
                    self.assertEqual(row["recovery_tokens_first_answer"], len(tokens) + answer_lengths[0] + 1)
                    self.assertLessEqual(row["recovery_tokens_first_answer"], 512)
                else:
                    original = raw_contexts[(row["source"], row["paragraph_id"])]
                    tokens = tokenizer.encode(original, add_special_tokens=False)
                    self.assertEqual(len(tokens), row["original_token_count"])
                    self.assertEqual(row["token_ids"], tokens[:len(row["token_ids"])])
                    self.assertEqual(row["text"], tokenizer.decode(row["token_ids"]))


if __name__ == "__main__":
    unittest.main()
