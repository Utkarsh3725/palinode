"""`compaction.md` v5: one op can retire a dated range, and v4's rules survive.

v4 made a later observation the PROPOSE_CONTRADICTS case and required RETRACT
to cite its evidence; those rules are pinned here unchanged, because a prompt
edit that quietly reverts them is the failure this file exists to catch.

What v5 adds is `ARCHIVE_BEFORE`. On the dogfood store the weekly pass was
shown 449 facts, nearly all of them stale dated session lines, and doing what
rule 4 asks — archive aggressively for status — meant ~430 operations with a
rationale each: ~60 KB, past any cap the model will serve, so the pass failed
and nothing was retired. One op naming a date retires the run, and the
proposal's size follows the number of reasons instead of the number of facts.
The rule-numbering and the v4 wording of rules 5, 8 and 9 are unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import frontmatter

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROMPT = REPO_ROOT / "specs" / "prompts" / "compaction.md"


def _text() -> str:
    return SOURCE_PROMPT.read_text(encoding="utf-8")


def _rule(number: int) -> str:
    rules = _text().split("## Rules", 1)[1].split("## Output Format", 1)[0]
    match = re.search(rf"^{number}\. (.*?)(?=^\d+\. |\Z)", rules, re.MULTILINE | re.DOTALL)
    assert match, f"rule {number} is missing"
    return " ".join(match.group(1).split())


def _ops_table_row(op: str) -> str:
    for line in _text().splitlines():
        if line.startswith(f"| {op} |"):
            return line
    raise AssertionError(f"no Operations row for {op}")


def _output_example() -> list[dict]:
    body = _text().split("## Output Format", 1)[1]
    blocks = re.findall(r"```json\n(.*?)```", body, re.DOTALL)
    return json.loads(blocks[0])


def test_prompt_declares_version_5() -> None:
    assert frontmatter.load(SOURCE_PROMPT).metadata["version"] == 5


def test_op_vocabulary_adds_only_the_range_op() -> None:
    kinds = {op["op"] for op in _output_example()}
    assert kinds == {
        "UPDATE", "MERGE", "SUPERSEDE", "ARCHIVE", "ARCHIVE_BEFORE", "RETRACT",
        "PROPOSE_CONTRADICTS",
    }
    assert '"KEEP"' not in json.dumps(_output_example())


def test_archive_before_row_and_example_carry_the_date_field() -> None:
    """The one field the executor reads. A range op without it is dropped."""
    row = _ops_table_row("ARCHIVE_BEFORE")
    assert "`before`" in row or "before" in row
    example = next(op for op in _output_example() if op["op"] == "ARCHIVE_BEFORE")
    assert example["before"] == "YYYY-MM-DD"
    assert example["reason"], "the range op must carry a rationale like every other"


def test_rule_4_prefers_the_range_op_for_a_run_of_dated_lines() -> None:
    """Naming the op in the table is not enough — rule 4 is where the model is
    told *when* to reach for it, and why one-op-per-line fails."""
    rule = _rule(4)
    assert "ARCHIVE_BEFORE" in rule
    assert "truncated" in rule


def test_rule_4_says_policy_retired_lines_are_already_gone() -> None:
    """Without this the model looks for lines the runner already retired and
    proposes a range that matches nothing."""
    rule = _rule(4)
    assert "retention policy" in rule
    assert "not in EXISTING_FACTS" in rule


def test_implicit_keep_contract_survives() -> None:
    text = _text()
    assert "Every fact you do not name is kept" in text
    assert "Emit an\noperation only for a fact you are changing" in text
    assert "is accepted and does exactly nothing" in text


def test_rules_are_still_numbered_one_through_ten() -> None:
    rules = _text().split("## Rules", 1)[1].split("## Output Format", 1)[0]
    assert re.findall(r"^(\d+)\. ", rules, re.MULTILINE) == [str(n) for n in range(1, 11)]


def test_rule_9_names_a_later_observation_as_the_contradicts_case() -> None:
    rule = _rule(9)
    assert "observed later" in rule
    assert "not rule 8" in rule
    assert "retires nothing" in rule


def test_rule_8_keeps_the_decision_for_action_without_retiring_anything() -> None:
    rule = _rule(8)
    assert "wins *for action*" in rule
    assert "Never SUPERSEDE, ARCHIVE or RETRACT a fact *because* a decision disagrees" in rule
    assert "KEEP it" in rule, "the v3 KEEP wording rule 8 relies on"


def test_retract_requires_a_citation_in_context() -> None:
    rule = _rule(5)
    assert "falsified_by" in rule
    assert '"Known to be incorrect" with no citation is forbidden' in rule
    assert "downgraded to PROPOSE_CONTRADICTS" in rule


def test_retract_row_and_example_carry_falsified_by() -> None:
    assert "falsified_by" in _ops_table_row("RETRACT")
    retract = next(op for op in _output_example() if op["op"] == "RETRACT")
    assert retract["falsified_by"] == "category/slug"
    assert "category/slug" in retract["reason"]


def test_rule_10_covers_both_ref_fields() -> None:
    rule = _rule(10)
    assert "`contradicts` and `falsified_by`" in rule
    assert "ACTIVE_DECISIONS or RECENT_NOTES" in rule
