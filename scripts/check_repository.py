"""Small dependency-free checks for the repository's maintained documentation."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def heading_ids(text: str) -> set[str]:
    ids: set[str] = set()
    counts: dict[str, int] = {}
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if fenced:
            continue
        match = re.match(r"^#{1,6}\s+(.+)", line)
        if not match:
            continue
        slug = re.sub(r"[`*_]", "", match[1]).lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"\s", "-", slug)
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        ids.add(f"{slug}-{count}" if count else slug)
    ids.update(re.findall(r'<a\s+id="([^"]+)"', text))
    return ids


def main() -> int:
    files = [ROOT / p for p in ("README.md", "README.en.md", "CONTRIBUTING.md", "AGENTS.md")]
    files.extend(sorted((ROOT / "docs").glob("*.md")))
    errors: list[str] = []
    links = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        label = path.relative_to(ROOT).as_posix()
        if "\ufffd" in text:
            errors.append(f"{label}: invalid replacement character")
        if sum(line.startswith("```") for line in text.splitlines()) % 2:
            errors.append(f"{label}: unclosed code fence")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            filename, separator, fragment = target.partition("#")
            linked = path.parent / unquote(filename) if filename else path
            links += 1
            if not linked.exists():
                errors.append(f"{label}: missing link {target}")
            elif separator and linked.suffix.lower() == ".md":
                if unquote(fragment) not in heading_ids(linked.read_text(encoding="utf-8")):
                    errors.append(f"{label}: missing anchor {target}")
    for path in (ROOT / "docs").glob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Repository checks passed: {len(files)} Markdown files, {links} local links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
