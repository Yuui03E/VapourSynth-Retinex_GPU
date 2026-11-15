"""Check public source syntax, JSON, local Markdown links and path hygiene."""
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("src", "python", "examples", "scripts", "tests", "docs", "licenses")


def main() -> None:
    files = [p for p in ROOT.iterdir() if p.is_file()]
    for folder in FOLDERS:
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.append(ROOT / "dist/README.md")
    failures = []
    for path in files:
        if path.suffix.lower() in (".pyc", ".zip", ".dll", ".png", ".jpg", ".jpeg", ".webp"):
            continue
        text = path.read_text(encoding="utf-8-sig")
        try:
            if path.suffix in (".py", ".vpy"):
                ast.parse(text, filename=str(path.relative_to(ROOT)))
            elif path.suffix == ".json":
                def reject_nonfinite(value):
                    raise ValueError(f"nonfinite JSON value: {value}")
                json.loads(text, parse_constant=reject_nonfinite)
        except (SyntaxError, ValueError) as error:
            failures.append(f"{path.relative_to(ROOT)}: {error}")
        if re.search(r"[A-Z]:[\\/]Users[\\/][^\\/\s<>]+", text, flags=re.I):
            failures.append(f"{path.relative_to(ROOT)}: contains a personal Windows home path")
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                if re.match(r"[a-z]+:|#", target, re.I):
                    continue
                target = unquote(target.split("#", 1)[0].strip("<>"))
                if target and not (path.parent / target).exists():
                    failures.append(f"{path.relative_to(ROOT)}: broken link {target}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Checked {len(files)} public files: syntax, JSON, links and personal paths passed.")


if __name__ == "__main__":
    main()
