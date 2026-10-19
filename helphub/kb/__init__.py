"""Customer-facing knowledge base: one markdown article per file, `# Title` on the first line."""

from pathlib import Path

from pydantic import BaseModel

KB_DIR = Path(__file__).parent


class Article(BaseModel):
    id: str  # file stem, e.g. "refund-policy"
    title: str
    text: str


def load_articles(directory: Path = KB_DIR) -> list[Article]:
    articles = []
    for path in sorted(directory.glob("*.md")):
        title, _, body = path.read_text(encoding="utf-8").partition("\n")
        articles.append(
            Article(id=path.stem, title=title.removeprefix("# ").strip(), text=body.strip())
        )
    return articles
