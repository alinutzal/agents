import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lib.telemetry_ablation import (  # noqa: E402
    apply_condition,
    candidate_agents,
    canonical_agent_name,
    is_tool_observation,
    select_indices,
    speaker_name,
)


SAMPLE = {
    "question": "Find the answer",
    "ground_truth": "42",
    "system_prompt": {"Researcher": "Use tools and report carefully."},
    "mistake_agent": "Researcher",
    "mistake_step": "3",
    "history": [
        {"role": "system", "content": "Coordinate the team."},
        {"role": "assistant", "name": "Researcher", "content": "I will inspect it."},
        {"role": "user", "name": "Computer_terminal", "content": "exitcode: 0\nCode output: 41"},
        {"role": "assistant", "name": "Researcher", "content": "The answer is 41."},
        {"role": "assistant", "name": "Critic", "content": "Looks good."},
    ],
}


class TelemetryClassificationTests(unittest.TestCase):
    def test_speaker_name_supports_both_who_when_subsets(self):
        self.assertEqual(speaker_name({"name": "Excel_Expert", "role": "assistant"}), "Excel_Expert")
        self.assertEqual(speaker_name({"role": "WebSurfer"}), "WebSurfer")

    def test_handcrafted_role_suffixes_are_canonicalized(self):
        self.assertEqual(canonical_agent_name("Orchestrator (thought)"), "Orchestrator")
        self.assertEqual(canonical_agent_name("Orchestrator (-> WebSurfer)"), "Orchestrator")
        self.assertEqual(canonical_agent_name("**Orchestrator** (-> WebSurfer)"), "Orchestrator")
        history = [
            {"role": "Orchestrator (thought)", "content": "plan"},
            {"role": "Orchestrator (-> WebSurfer)", "content": "search"},
            {"role": "WebSurfer", "content": "result"},
        ]
        self.assertEqual(candidate_agents(history), ["Orchestrator", "WebSurfer"])
        self.assertEqual(
            select_indices(history, "oracle_responsible_agent", mistake_agent="Orchestrator"),
            [0, 1],
        )

    def test_tool_observation_uses_role_name_and_content(self):
        self.assertTrue(is_tool_observation({"role": "tool", "content": "ok"}))
        self.assertTrue(is_tool_observation(SAMPLE["history"][2]))
        self.assertTrue(is_tool_observation({"role": "user", "content": "Observation: page loaded"}))
        self.assertFalse(is_tool_observation({"role": "WebSurfer", "content": "I searched the web"}))


class TelemetryConditionTests(unittest.TestCase):
    def test_default_views_select_expected_steps(self):
        self.assertEqual(select_indices(SAMPLE["history"], "full"), [0, 1, 2, 3, 4])
        self.assertEqual(select_indices(SAMPLE["history"], "no_system_messages"), [1, 2, 3, 4])
        self.assertEqual(select_indices(SAMPLE["history"], "no_tool_observations"), [0, 1, 3, 4])
        self.assertEqual(select_indices(SAMPLE["history"], "tool_observations_only"), [2])
        self.assertEqual(select_indices(SAMPLE["history"], "communication_only"), [1, 3, 4])
        self.assertEqual(
            select_indices(SAMPLE["history"], "oracle_responsible_agent", mistake_agent="Researcher"),
            [1, 3],
        )

    def test_mapping_converts_local_prediction_back_to_original_step(self):
        result = apply_condition(SAMPLE, "communication_only", trace_id="sample.json")
        self.assertEqual(result.retained_indices, [1, 3, 4])
        self.assertEqual(result.local_to_original, {"0": 1, "1": 3, "2": 4})
        self.assertEqual(result.original_to_local("3"), 1)
        self.assertIsNone(result.original_to_local("2"))

    def test_random_control_is_trace_stable_and_seeded(self):
        first = select_indices(SAMPLE["history"], "random_half", trace_id="1.json", seed=7)
        second = select_indices(SAMPLE["history"], "random_half", trace_id="1.json", seed=7)
        different = select_indices(SAMPLE["history"], "random_half", trace_id="1.json", seed=8)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertNotEqual(first, different)

    def test_transformation_does_not_mutate_source(self):
        before = json.dumps(SAMPLE, sort_keys=True)
        apply_condition(SAMPLE, "recent_half", trace_id="sample.json")
        self.assertEqual(json.dumps(SAMPLE, sort_keys=True), before)

    def test_unknown_condition_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "Unknown telemetry condition"):
            select_indices(SAMPLE["history"], "not_a_condition")


class BuilderIntegrationTests(unittest.TestCase):
    def test_builder_writes_views_and_original_step_mapping(self):
        # Importing the script's pure builder function avoids invoking a model.
        scripts_root = REPO_ROOT / "scripts"
        if str(scripts_root) not in sys.path:
            sys.path.insert(0, str(scripts_root))
        from build_telemetry_ablations import build_condition_record

        output, row = build_condition_record(SAMPLE, "communication_only", "sample.json", 2026)
        self.assertEqual(len(output["history"]), 3)
        self.assertEqual(row["gold_original_step"], "3")
        self.assertTrue(row["gold_step_valid"])
        self.assertEqual(row["gold_local_step"], 1)
        self.assertEqual(
            output["candidate_agents"], ["Researcher", "Computer_terminal", "Critic"]
        )
        self.assertTrue(row["gold_step_visible"])
        self.assertEqual(row["local_to_original"]["1"], 3)

        without_answer, _ = build_condition_record(SAMPLE, "no_reference_answer", "sample.json", 2026)
        self.assertNotIn("ground_truth", without_answer)
        _, no_answer_row = build_condition_record(SAMPLE, "no_reference_answer", "sample.json", 2026)
        self.assertFalse(no_answer_row["view_identical_to_full"])

        without_system, system_row = build_condition_record(
            SAMPLE, "no_system_messages", "sample.json", 2026
        )
        self.assertNotIn("system_prompt", without_system)
        self.assertLess(system_row["retained_characters"], system_row["original_characters"])


if __name__ == "__main__":
    unittest.main()
