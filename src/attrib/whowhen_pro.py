"""Who&When Pro text traces: rendering, label resolution, and WHO/WHEN/WHAT option sets.

Every framework stores its label in its own coordinates. `build_item` resolves them to
trajectory indices, the same indices the rendered text shows as "[Step i ...]":

  alfagent, smolagents   single agent; `step` matches the trajectory's `step_number`
                         (smolagents step 0 is the planning step). WHO does not apply.
  magentic-one           `step` is "round.position"
  mathchat               `round` + `position`
  macnet                 `position` indexes the labelled agent's own steps; `accepted_predictions`
                         lists alternate (agent, step, mode) answers that also count as correct
  metagpt                `stage` is the index among the agent steps
  debate, dylan          `round`; `agents` lists 1-2 responsible agents, any of which is correct

WHEN options are every trajectory step except the task prompt (`kind == "user"`) and the final
answer (`kind == "final_answer"`). Correctness is membership in the accepted set, so random
guessing scores |accepted| / |options| per trace.
"""
import json
import os

import yaml

# Fields that reveal where the error was injected. They must never reach the auditor's input.
LEAKY_FIELDS = {"is_injected", "is_replayed", "is_live", "phase", "injection_via"}
# Metadata that is not part of what an agent said or saw.
SKIPPED_FIELDS = LEAKY_FIELDS | {"kind", "observation_images", "is_final_answer"}
# Keys used to name who produced a step, in order of preference.
SPEAKER_FIELDS = ("agent", "agent_id", "role")
# Keys whose values are only used in the step header.
HEADER_FIELDS = set(SPEAKER_FIELDS) | {"step_number", "round", "position", "stage"}
NON_CANDIDATE_KINDS = {"user", "final_answer"}

SINGLE_AGENT_FRAMEWORKS = {"alfagent", "smolagents"}
# Length strata (Laya tokens of the rendered trace) for sampling and C1.3.
LONG_CUTOFF = 3000
CONTEXT_LIMIT = 8192

WHO_INSTRUCTIONS = ("Which agent made the mistake that is directly responsible for the wrong "
                    "solution to the task?")
WHEN_INSTRUCTIONS = ("At which step did the first decisive mistake that led to the wrong solution "
                     "happen?")
WHAT_INSTRUCTIONS = "Which kind of error is the decisive mistake that led to the wrong solution?"
NONE_OPTION = "none"
NONE_TEXT = "none of these steps; the decisive mistake is not in this part of the conversation"


def as_obj(value):
    """Some columns are stored as JSON strings; decode them if so."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _fmt(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def speaker_of(step):
    if not isinstance(step, dict):
        return "unknown"
    speaker = next((step[k] for k in SPEAKER_FIELDS if step.get(k) not in (None, "")), None)
    return speaker if speaker is not None else step.get("kind", "unknown")


def _step_header(step, idx):
    parts = [f"Step {idx}"]
    for key in ("step_number", "round", "position", "stage"):
        if step.get(key) is not None:
            parts.append(f"{key}={step[key]}")
    return f"[{' '.join(parts)}] {speaker_of(step)}"


def _render_fields(obj):
    lines = []
    for key, value in obj.items():
        if key in SKIPPED_FIELDS or key in HEADER_FIELDS or key == "turns":
            continue
        if value is None or value == "" or value == []:
            continue
        if key == "content":
            lines.append(_fmt(value))
        else:
            lines.append(f"{key}: {_fmt(value)}")
    return lines


def render_parts(row, include_answer=False):
    """The task header lines and one list of lines per trajectory step."""
    task = as_obj(row.get("task")) or {}
    trajectory = as_obj(row.get("trajectory")) or []

    header = []
    if isinstance(task, dict):
        header.append(f"Task: {_fmt(task.get('query', ''))}")
        if include_answer and task.get("answer") is not None:
            header.append(f"Answer: {_fmt(task['answer'])}")
    else:
        header.append(f"Task: {_fmt(task)}")

    blocks = []
    for idx, step in enumerate(trajectory):
        if not isinstance(step, dict):
            blocks.append([f"[Step {idx}] {_fmt(step)}"])
            continue
        lines = [_step_header(step, idx)]
        lines.extend(_render_fields(step))
        # Debate/DyLAN rounds hold one turn per agent.
        for turn in as_obj(step.get("turns")) or []:
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("agent_id", "unknown")
            if turn.get("role"):
                speaker = f"{speaker} ({turn['role']})"
            lines.append(f"  - {speaker}: {_fmt(turn.get('output', ''))}")
        lines.append("")
        blocks.append(lines)
    return header, blocks


def render_trace(row, include_answer=False):
    """Render one trace as plain text, excluding label-leaking fields and extras."""
    header, blocks = render_parts(row, include_answer)
    return "\n".join(header + [line for block in blocks for line in block])


def load_taxonomy(path="data/whowhen_pro/taxonomy.yaml", text_only=True):
    """Error modes as {code: {"name", "description"}}; perception modes need images."""
    with open(path, encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    return {code: v for code, v in tax.items() if not (text_only and code.startswith("P."))}


def mode_family(code):
    return code.split(".")[0]


def length_stratum(n_tokens):
    if n_tokens > CONTEXT_LIMIT:
        return "overlong"
    return "long" if n_tokens >= LONG_CUTOFF else "short"


def _agent_options(row, trajectory):
    extras = as_obj(row.get("extras")) or {}
    agents = extras.get("agents") if isinstance(extras, dict) else None
    if agents:
        return {a["id"]: f"the agent {a['id']}" + (f" ({a['role']})" if a.get("role") and a["role"] != a["id"] else "")
                for a in agents}
    speakers = []
    for step in trajectory:
        if isinstance(step, dict) and step.get("kind") not in NON_CANDIDATE_KINDS:
            for turn in as_obj(step.get("turns")) or []:
                speakers.append(turn.get("agent_id", "unknown"))
            if not step.get("turns"):
                speakers.append(speaker_of(step))
    return {s: f"the agent {s}" for s in dict.fromkeys(speakers)}


def _nth_step_of(trajectory, agent, n):
    mine = [i for i, s in enumerate(trajectory) if isinstance(s, dict) and s.get("agent_id") == agent]
    return mine[n] if n < len(mine) else None


def resolve_labels(framework, gt, trajectory):
    """(accepted agents or None, accepted WHEN indices, accepted modes) for one trace."""
    steps = [s if isinstance(s, dict) else {} for s in trajectory]
    modes = {gt["mode"]}
    agents, when = None, set()
    if framework in SINGLE_AGENT_FRAMEWORKS:
        when = {i for i, s in enumerate(steps) if s.get("step_number") == gt["step"] and s.get("kind") != "user"}
        if gt["step"] == 0:  # smolagents planning step carries no step_number
            when = {i for i, s in enumerate(steps) if s.get("kind") == "planning"}
    elif framework == "magentic-one":
        rd, pos = (int(x) for x in str(gt["step"]).split("."))
        when = {i for i, s in enumerate(steps) if s.get("round") == rd and s.get("position") == pos}
        agents = {gt["agent"]}
    elif framework == "mathchat":
        when = {i for i, s in enumerate(steps) if s.get("round") == gt["round"] and s.get("position") == gt["position"]}
        agents = {gt["agent"]}
    elif framework == "macnet":
        agents = {gt["agent"]}
        when = {_nth_step_of(trajectory, gt["agent"], gt["position"])}
        for alt in gt.get("accepted_predictions") or []:
            _, pos = (int(x) for x in alt["step_coord"].split("."))
            agents.add(alt["agent_name"])
            when.add(_nth_step_of(trajectory, alt["agent_name"], pos))
            modes.add(alt["mode"])
        when.discard(None)
    elif framework == "metagpt":
        agent_steps = [i for i, s in enumerate(steps) if s.get("agent_id")]
        when = {agent_steps[gt["stage"]]}
        agents = {gt["agent"]}
    elif framework in ("debate", "dylan"):
        when = {i for i, s in enumerate(steps) if s.get("kind") == "round" and s.get("round") == gt["round"]}
        agents = set(gt["agents"])
    else:
        raise ValueError(f"unknown framework {framework!r}")
    return agents, when, modes


def build_item(row, taxonomy):
    """One trace as the model-agnostic unit every experiment consumes."""
    trajectory = as_obj(row["trajectory"]) or []
    gt = as_obj(row["ground_truth"])
    framework = row["framework"]
    agents, when, modes = resolve_labels(framework, gt, trajectory)

    candidates = [i for i, s in enumerate(trajectory)
                  if not (isinstance(s, dict) and s.get("kind") in NON_CANDIDATE_KINDS)]
    when_options = {}
    for i in candidates:
        step = trajectory[i] if isinstance(trajectory[i], dict) else {}
        if step.get("turns"):
            when_options[str(i)] = f"step {i}, round {step.get('round', i)} of the discussion"
        else:
            when_options[str(i)] = f"step {i}, by {speaker_of(step)}"

    who_options = None
    if framework not in SINGLE_AGENT_FRAMEWORKS:
        who_options = _agent_options(row, trajectory)

    n_tokens = row.get("n_tokens")
    return {
        "id": row["id"],
        "framework": framework,
        "benchmark": row["benchmark"],
        "n_tokens": n_tokens,
        "stratum": length_stratum(n_tokens) if n_tokens is not None else None,
        "n_steps": len(candidates),
        "n_agents": len(who_options) if who_options else 1,
        "who_options": who_options,
        "when_options": when_options,
        "what_options": {code: v["name"] for code, v in taxonomy.items()},
        "who_true": sorted(agents) if agents is not None else None,
        "when_true": sorted(str(i) for i in when),
        "what_true": sorted(modes),
        "what_canonical": gt["mode"],
    }


def check_item(item):
    """Problems that would make a question unanswerable; empty when the item is usable."""
    problems = []
    if not item["when_true"] or not set(item["when_true"]) <= set(item["when_options"]):
        problems.append("when label outside options")
    if item["who_options"] is not None and not set(item["who_true"]) & set(item["who_options"]):
        problems.append("who label outside options")
    if not set(item["what_true"]) & set(item["what_options"]):
        problems.append("what label outside options")
    return problems


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_items(path, taxonomy_path="data/whowhen_pro/taxonomy.yaml"):
    """Rows of a filtered jsonl (with `rendered_text` and `n_tokens`) as (row, item) pairs."""
    taxonomy = load_taxonomy(taxonomy_path)
    return [(row, build_item(row, taxonomy)) for row in load_rows(path)]


def default_pool_path(pool, data_dir="data/whowhen_pro", max_len=CONTEXT_LIMIT):
    name = {"fit": f"text_ctx{max_len}.jsonl", "overlong": f"text_over{max_len}.jsonl"}[pool]
    return os.path.join(data_dir, name)
