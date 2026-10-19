import pytest
from fakes import FakeOpenAI, case_ids, load_cases

from helphub.kb import Article
from helphub.qa.rag_answer import RAG_SYSTEM_PROMPT, answer_question

CASES = load_cases("questions")
REFUND = Article(id="refund-policy", title="Refund policy", text="30 days.")
CANCEL = Article(id="cancel-subscription", title="Cancelling", text="Any time.")


def test_sources_are_the_excerpts_the_answer_cites():
    client = FakeOpenAI("Within 30 days [refund-policy]. Cancel any time [cancel-subscription].")

    answer = answer_question(client, "refund?", [CANCEL, REFUND])

    assert answer.sources == ["cancel-subscription", "refund-policy"]
    assert answer.text.startswith("Within 30 days")


def test_an_uncited_or_empty_reply_has_no_sources():
    assert answer_question(FakeOpenAI("I do not know."), "?", [REFUND]).sources == []
    assert answer_question(FakeOpenAI(None), "?", [REFUND]).text == ""


def test_request_contract():
    client = FakeOpenAI("ok")

    answer_question(client, "Can I return it?", [REFUND, CANCEL])

    (call,) = client.calls
    assert call["model"] == "gpt-4o"
    assert call["messages"][0] == {"role": "system", "content": RAG_SYSTEM_PROMPT}
    user = call["messages"][1]["content"]
    assert "[refund-policy] Refund policy\n30 days." in user
    assert "[cancel-subscription] Cancelling\nAny time." in user
    assert user.endswith("Customer question: Can I return it?")


def test_no_articles_is_said_explicitly():
    client = FakeOpenAI("ok")

    answer_question(client, "Gift cards?", [])

    assert "(none found)" in client.calls[0]["messages"][1]["content"]


@pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
def test_every_fixture_question_is_sent_with_its_snippets(case):
    client = FakeOpenAI("An answer.")
    articles = [Article(**s) for s in case["snippets"]]

    answer_question(client, case["question"], articles)

    user = client.calls[0]["messages"][1]["content"]
    assert case["question"] in user
    for article in articles:
        assert f"[{article.id}] {article.title}" in user
