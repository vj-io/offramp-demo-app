import re

from helphub.kb import load_articles


def test_the_shipped_knowledge_base_is_small_and_well_formed():
    articles = load_articles()

    assert 10 <= len(articles) <= 20
    assert len({a.id for a in articles}) == len(articles)
    for article in articles:
        assert article.title and not article.title.startswith("#")
        assert 20 <= len(article.text.split()) <= 150, article.id


def test_loader_reads_title_and_body(tmp_path):
    (tmp_path / "hello-world.md").write_text("# Hello\n\nFirst para.\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("nope", encoding="utf-8")

    (article,) = load_articles(tmp_path)

    assert (article.id, article.title, article.text) == ("hello-world", "Hello", "First para.")


def test_every_delivery_window_is_stated_as_the_same_number_of_days():
    windows = {
        days
        for article in load_articles()
        for days in re.findall(r"within (\d+) days of delivery", article.text)
    }

    assert windows == {"30"}
