import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_telemetry_experiment.py"


def load_runner():
    # The parser/resume helpers are pure; lightweight stubs keep unit tests CPU-only.
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.bfloat16 = "bfloat16"
    torch.float32 = "float32"
    sys.modules.setdefault("torch", torch)
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    transformers.Pipeline = object
    transformers.pipeline = lambda *args, **kwargs: None
    sys.modules.setdefault("transformers", transformers)
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **kwargs: iterable
    sys.modules.setdefault("tqdm", tqdm_module)
    spec = importlib.util.spec_from_file_location("telemetry_runner", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_parses_markdown_labels_and_values(self):
        agent, step = self.runner.parse_prediction(
            "**Agent Name:** **Orchestrator**\n**Step Number:** **7**\nReason for Mistake: x"
        )
        self.assertEqual(agent, "Orchestrator")
        self.assertEqual(step, 7)

    def test_canonicalizes_handcrafted_agent_variant(self):
        agent, step = self.runner.parse_prediction(
            "Agent Name: **Orchestrator** (-> WebSurfer)\nStep Number: 2"
        )
        self.assertEqual(agent, "Orchestrator")
        self.assertEqual(step, 2)

    def test_resume_rejects_mismatched_run(self):
        path = REPO_ROOT / ".test_telemetry_runner_results.jsonl"
        try:
            path.write_text(
                '{"run_id":"old","condition":"full","trace_id":"1.json"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "belongs to run"):
                self.runner.existing_keys(path, "new")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
