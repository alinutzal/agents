"""Pure utilities for controlled telemetry ablations on Who&When traces.

The original Who&When JSON files contain a top-level ``history`` list.  This
module filters that list while retaining a mapping from the condition-local
step numbers shown to an attribution model back to the original step numbers.
It deliberately has no model or dataset dependencies so the transformations
can be tested before expensive inference is launched.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DEFAULT_CONDITIONS = (
    "full",
    "no_system_messages",
    "no_tool_observations",
    "tool_observations_only",
    "communication_only",
    "oracle_responsible_agent",
)

# Conservative names for execution environments.  WebSurfer and FileSurfer
# are intentionally absent: in the hand-crafted data they are autonomous
# agents rather than passive tool-result channels.
_TERMINAL_NAMES = {
    "computerterminal",
    "computer_terminal",
    "computer-terminal",
    "terminal",
    "shell",
    "python_terminal",
}
_TOOL_RESULT_PREFIX = re.compile(
    r"^\s*(?:exitcode\s*:|code output\s*:|tool (?:result|output)\s*:|observation\s*:)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AblationResult:
    """A filtered trace and the information needed to score it fairly."""

    history: List[Dict[str, Any]]
    retained_indices: List[int]
    original_step_count: int

    @property
    def local_to_original(self) -> Dict[str, int]:
        return {str(local): original for local, original in enumerate(self.retained_indices)}

    def original_to_local(self, original_step: Any) -> int | None:
        try:
            step = int(original_step)
        except (TypeError, ValueError):
            return None
        try:
            return self.retained_indices.index(step)
        except ValueError:
            return None


def speaker_name(entry: Mapping[str, Any]) -> str:
    """Return the best available speaker label for either dataset subset."""

    for key in ("name", "agent", "agent_id", "role"):
        value = entry.get(key)
        if value not in (None, ""):
            return str(value)
    return "Unknown Agent"


def is_system_message(entry: Mapping[str, Any]) -> bool:
    return str(entry.get("role", "")).strip().casefold() == "system"


def is_tool_observation(entry: Mapping[str, Any]) -> bool:
    """Conservatively identify passive tool/terminal result messages.

    Who&When predates a single tool-event schema.  The classifier therefore
    uses explicit tool roles/IDs first, then terminal agent names and common
    execution-result prefixes.  It does not classify WebSurfer/FileSurfer as
    tools because those are decision-making agents in the hand-crafted set.
    """

    role = str(entry.get("role", "")).strip().casefold()
    if role in {"tool", "function"} or entry.get("tool_call_id") is not None:
        return True

    normalized_name = speaker_name(entry).strip().casefold().replace(" ", "_")
    if normalized_name in _TERMINAL_NAMES:
        return True

    content = entry.get("content", "")
    if isinstance(content, (dict, list)):
        content = json.dumps(content, ensure_ascii=False)
    return bool(_TOOL_RESULT_PREFIX.match(str(content)))


def _stable_seed(seed: int, trace_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{trace_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def select_indices(
    history: Sequence[Mapping[str, Any]],
    condition: str,
    *,
    mistake_agent: Any = None,
    trace_id: str = "",
    seed: int = 0,
) -> List[int]:
    """Select original history indices for one named ablation condition."""

    n_steps = len(history)
    all_indices = list(range(n_steps))

    if condition in {"full", "no_reference_answer"}:
        return all_indices
    if condition == "no_system_messages":
        return [i for i, entry in enumerate(history) if not is_system_message(entry)]
    if condition == "no_tool_observations":
        return [i for i, entry in enumerate(history) if not is_tool_observation(entry)]
    if condition == "tool_observations_only":
        return [i for i, entry in enumerate(history) if is_tool_observation(entry)]
    if condition == "communication_only":
        return [
            i
            for i, entry in enumerate(history)
            if not is_system_message(entry) and not is_tool_observation(entry)
        ]
    if condition == "oracle_responsible_agent":
        target = str(mistake_agent).strip().casefold()
        if not target:
            return []
        return [
            i for i, entry in enumerate(history)
            if speaker_name(entry).strip().casefold() == target
        ]
    if condition in {"first_half", "recent_half", "random_half"}:
        keep = int(math.ceil(n_steps / 2))
        if condition == "first_half":
            return all_indices[:keep]
        if condition == "recent_half":
            return all_indices[-keep:]
        rng = random.Random(_stable_seed(seed, trace_id))
        return sorted(rng.sample(all_indices, keep)) if keep else []

    raise ValueError(f"Unknown telemetry condition: {condition}")


def apply_condition(
    record: Mapping[str, Any],
    condition: str,
    *,
    trace_id: str = "",
    seed: int = 0,
) -> AblationResult:
    """Apply a condition without mutating the source record."""

    raw_history = record.get("history", [])
    if not isinstance(raw_history, list):
        raise ValueError("record['history'] must be a list")

    indices = select_indices(
        raw_history,
        condition,
        mistake_agent=record.get("mistake_agent"),
        trace_id=trace_id,
        seed=seed,
    )
    history = [dict(raw_history[i]) for i in indices]
    return AblationResult(history, indices, len(raw_history))


def serialized_history_size(
    history: Iterable[Mapping[str, Any]], system_prompt: Any = None
) -> Dict[str, int]:
    """Return dependency-free telemetry-size proxies used in cost analysis."""

    payload: Dict[str, Any] = {"history": list(history)}
    if system_prompt not in (None, "", {}, []):
        payload["system_prompt"] = system_prompt
    rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return {
        "characters": len(rendered),
        # This is explicitly a whitespace-token proxy, not a model token count.
        "word_tokens": len(rendered.split()),
    }
