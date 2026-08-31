from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class _HTMLInventory(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids.append(element_id)
        if tag == "a" and attributes.get("href"):
            self.hrefs.append(str(attributes["href"]))


def _inventory(path: Path) -> _HTMLInventory:
    parser = _HTMLInventory()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def test_html_documentation_has_unique_ids_and_resolvable_local_links() -> None:
    pages = (DOCS / "index.html", DOCS / "installation.html")
    inventories = {page.resolve(): _inventory(page) for page in pages}
    for page, inventory in inventories.items():
        assert len(inventory.ids) == len(set(inventory.ids)), page
        for href in inventory.hrefs:
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc:
                continue
            target = (
                page.parent / unquote(parsed.path)
                if parsed.path
                else page
            ).resolve()
            assert target.exists(), f"{page}: broken link {href!r}"
            if parsed.fragment and target.suffix.casefold() == ".html":
                target_inventory = inventories.get(target) or _inventory(target)
                assert parsed.fragment in target_inventory.ids, (
                    f"{page}: missing fragment target {href!r}"
                )


def test_installation_page_covers_both_checked_backends() -> None:
    source = (DOCS / "installation.html").read_text(encoding="utf-8")
    required = (
        'id="windows"',
        'id="linux"',
        "environment.yml",
        "phasefield-fenicsx",
        "scripts/verify_fenicsx_install.py",
        "linux_cluster/environment-linux.yml",
        "phasefield-fenicsx-linux",
        "linux_cluster/check_environment.py",
        "linux_cluster/run_linux.sh",
        "linux_cluster/run_xeon16_suite.sh",
        "FENICSX_INSTALL_OK",
        "PETSc",
        "MPI",
    )
    for token in required:
        assert token in source
    assert "D:\\Github" not in source
    assert "D:\\Software" not in source


def test_entry_documentation_links_to_installation_guide() -> None:
    index = (DOCS / "index.html").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "installation.html" in index
    assert "docs/installation.html" in readme
    assert "tmp\\verify_fenicsx_install.py" not in index
    assert "tmp/verify_fenicsx_install.py" not in readme
    assert "D:\\Github" not in index
    assert "D:\\Software" not in readme
    assert (ROOT / "scripts" / "verify_fenicsx_install.py").is_file()


def test_git_ignore_excludes_local_runtimes_but_keeps_curated_docs() -> None:
    source = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for token in (
        ".fenicsx-env/",
        ".micromamba-root/",
        ".tools/",
        "tmp/",
        "results/*",
        "!results/demo/load_displacement.png",
        "!output/pdf/PHASEFIELD_ALGORITHM_ACCEPTANCE_AND_CORRECTIONS.pdf",
    ):
        assert token in source
