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
