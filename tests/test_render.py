import json
from pathlib import Path

from sdr.models import Brief
from sdr.render import render_brief

REPO_ROOT = Path(__file__).parent.parent


def test_render_matches_example_brief():
    data = json.loads((REPO_ROOT / "fixtures" / "brief.json").read_text())
    brief = Brief.model_validate(data)

    rendered = render_brief(brief)
    expected = (REPO_ROOT / "docs" / "example_brief.md").read_text()

    assert rendered.strip() == expected.strip()
