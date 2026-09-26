"""Check completion supervision excludes prompts and never truncates answers."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("diagnostic_sft", Path(__file__).with_name("train_sft.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TokenizerStub:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return "P:" + messages[0]["content"] + ":A:"

    def encode(self, text, **kwargs):
        return [ord(x) for x in text]


class CompletionContract(unittest.TestCase):
    def test_only_answer_and_eos_supervised(self):
        row = module.encode_completion(TokenizerStub(), "forecast", "[2,3]", 100)
        prefix = row["prompt_length"]
        self.assertEqual(row["labels"][:prefix], [-100] * prefix)
        self.assertEqual(row["labels"][prefix:], [91, 50, 44, 51, 93, 99])
        self.assertEqual(len(row["labels"]), len(row["input_ids"]))

    def test_overflow_rejected_instead_of_answer_truncation(self):
        with self.assertRaises(ValueError):
            module.encode_completion(TokenizerStub(), "forecast", "[2,3]", 4)


if __name__ == "__main__":
    unittest.main()
