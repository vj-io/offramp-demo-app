"""Pins the shape of HelpHub's LLM call sites.

The demo is a target for Offramp's AST scanner, so each call site must stay a direct
`client.<api>.create(...)` call with literal keyword arguments, exactly one per file, with its
prompt held in a module-level constant. These tests fail if a refactor hides a call site.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parent.parent / "helphub"
API_CHAINS = {
    ("chat", "completions", "create"): "chat",
    ("messages", "create"): "chat",
    ("embeddings", "create"): "embeddings",
}

# file -> (model, module-level constant holding the prompt)
CHAT_SITES = {
    "inbox/triage.py": ("gpt-4o-mini", "TRIAGE_SYSTEM_PROMPT"),  # cs_01
    "ops/agent_step.py": ("claude-sonnet-4-5", "AGENT_SYSTEM_PROMPT"),  # cs_02
    "orders/extract.py": ("claude-sonnet-4-5", "EXTRACT_SYSTEM_PROMPT"),  # cs_03
    "qa/rag_answer.py": ("gpt-4o", "RAG_SYSTEM_PROMPT"),  # cs_04
    "refunds/plan.py": ("o3", "REFUND_POLICY"),  # cs_05
    "summaries/summarize.py": ("claude-haiku-4-5", "SUMMARY_SYSTEM_PROMPT"),  # cs_06
}


def attribute_chain(node: ast.expr) -> tuple[str, ...]:
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    return tuple(reversed(names))


def llm_calls(path: Path) -> list[tuple[str, ast.Call]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = attribute_chain(node.func)
            for suffix, kind in API_CHAINS.items():
                if chain[-len(suffix) :] == suffix:
                    found.append((kind, node))
    return found


def module_constants(path: Path) -> dict[str, ast.expr]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }


def keywords(call: ast.Call) -> dict[str, ast.expr]:
    assert not call.args, "call sites use keyword arguments only"
    assert all(k.arg for k in call.keywords), "no **kwargs unpacking at a call site"
    return {k.arg: k.value for k in call.keywords}


def test_there_are_exactly_these_llm_calls():
    per_file = {
        str(p.relative_to(PACKAGE)): [kind for kind, _ in llm_calls(p)]
        for p in sorted(PACKAGE.rglob("*.py"))
    }
    found = {path: kinds for path, kinds in per_file.items() if kinds}

    assert found == {**{path: ["chat"] for path in CHAT_SITES}, "search/embed.py": ["embeddings"]}


def test_chat_sites_sort_into_the_documented_scan_order():
    assert list(CHAT_SITES) == sorted(CHAT_SITES)


@pytest.mark.parametrize("relative", CHAT_SITES)
def test_chat_site_uses_a_literal_model_and_a_module_level_prompt(relative):
    model, prompt = CHAT_SITES[relative]
    path = PACKAGE / relative
    ((_, call),) = llm_calls(path)
    kwargs = keywords(call)

    assert isinstance(kwargs["model"], ast.Constant) and kwargs["model"].value == model
    assert isinstance(module_constants(path)[prompt], ast.Constant)
    assert prompt in ast.unparse(call)
    assert any(isinstance(n, ast.JoinedStr) for n in ast.walk(call)), "user message is an f-string"


def test_openai_reasoning_site_follows_the_o3_parameter_rules():
    ((_, call),) = llm_calls(PACKAGE / "refunds/plan.py")
    kwargs = keywords(call)

    assert "max_completion_tokens" in kwargs
    assert "temperature" not in kwargs and "max_tokens" not in kwargs


def test_tool_use_site_passes_a_module_level_tools_list():
    path = PACKAGE / "ops/agent_step.py"
    ((_, call),) = llm_calls(path)

    assert isinstance(keywords(call)["tools"], ast.Name)
    assert isinstance(module_constants(path)["TOOLS"], ast.List)


def test_json_extraction_site_builds_its_schema_from_a_pydantic_model():
    path = PACKAGE / "orders/extract.py"

    assert ast.unparse(module_constants(path)["ORDER_SCHEMA"]) == "OrderFields.model_json_schema()"
    ((_, call),) = llm_calls(path)
    assert "ORDER_SCHEMA" in ast.unparse(call)
