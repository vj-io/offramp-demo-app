import pytest
from fakes import FakeOpenAI

from helphub.kb import Article, load_articles
from helphub.search.embed import cosine, embed_texts, search_kb


def test_embed_texts_returns_vectors_in_input_order():
    client = FakeOpenAI(embed=lambda text: [float(len(text)), 0.0])

    assert embed_texts(client, ["a", "bbb", "cc"]) == [[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]]
    assert client.calls[0]["model"] == "text-embedding-3-small"
    assert client.calls[0]["input"] == ["a", "bbb", "cc"]


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


def test_search_returns_the_most_similar_articles_first():
    articles = load_articles()

    hits = search_kb(FakeOpenAI(), "forgot my password, send a reset link", articles, k=2)

    assert [a.id for a in hits][0] == "reset-password"
    assert len(hits) == 2


@pytest.mark.parametrize("query", ["", "   \n"])
def test_blank_query_does_not_call_the_api(query):
    client = FakeOpenAI()

    assert search_kb(client, query, [Article(id="a", title="A", text="a")]) == []
    assert client.calls == []


def test_no_articles_does_not_call_the_api():
    client = FakeOpenAI()

    assert search_kb(client, "refund", []) == []
    assert client.calls == []
