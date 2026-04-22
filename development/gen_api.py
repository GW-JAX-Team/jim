#!/usr/bin/env python3
"""Generate API reference stubs and produce _zensical_build.toml.

Run before: uv run zensical build -f _zensical_build.toml --clean
For local development: uv run zensical serve -f _zensical_build.toml

Environment variables:
    VERSION_ALIAS  When set (e.g. "stable", "latest", "development"),
                   appends the alias to site_url in the output config so
                   that versioned builds have correct canonical URLs.
"""
import os
import re
import shutil
import tomllib
from pathlib import Path

# Module path prefixes (relative to src/) to exclude from the API docs.
# Use forward slashes, e.g. "mypkg/internal".
SKIP_PREFIXES: list[str] = []


# ── module scanning ───────────────────────────────────────────────────────────

def scan_modules(src_dir: Path) -> list[tuple[list[str], Path]]:
    """Return (module_parts, stub_path) for every public Python module."""
    results = []
    for py_file in sorted(src_dir.rglob("*.py")):
        parts = list(py_file.relative_to(src_dir).with_suffix("").parts)
        if parts[-1] in ("__init__", "__main__"):
            continue
        rel = "/".join(parts)
        if any(rel.startswith(skip) for skip in SKIP_PREFIXES):
            continue
        stub_path = Path("api", *parts).with_suffix(".md")
        results.append((parts, stub_path))
    return results


def write_stubs(modules: list[tuple[list[str], Path]], docs_dir: Path) -> None:
    """Write (or overwrite) ::: identifier stubs under docs/api/."""
    for parts, stub_path in modules:
        full_path = docs_dir / stub_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"::: {'.'.join(parts)}\n")


# ── nav generation ────────────────────────────────────────────────────────────

def build_nav_tree(modules: list[tuple[list[str], Path]]) -> list:
    """Convert flat module list into a nested nav structure."""
    tree: dict = {}
    for parts, stub_path in modules:
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = stub_path.as_posix()

    def dict_to_nav(d: dict) -> list:
        return [
            {k: dict_to_nav(v) if isinstance(v, dict) else v}
            for k, v in d.items()
        ]

    return dict_to_nav(tree)


def nav_to_toml_str(nav: list, depth: int = 0) -> str:
    """Serialize a nav list to the TOML array-of-inline-tables format."""
    inner = "    " * (depth + 1)
    lines = ["["]
    for item in nav:
        if isinstance(item, str):
            # Bare path entry (no label), e.g. "tutorials/index.md"
            lines.append(f'{inner}"{item}",')
        else:
            for k, v in item.items():
                if isinstance(v, str):
                    lines.append(f'{inner}{{"{k}" = "{v}"}},')
                else:
                    nested = nav_to_toml_str(v, depth + 1)
                    lines.append(f'{inner}{{"{k}" = {nested}}},')
    lines.append("    " * depth + "]")
    return "\n".join(lines)


# ── TOML patching ─────────────────────────────────────────────────────────────

def replace_nav(toml_text: str, new_nav: list) -> str:
    """Replace the nav = [...] block in TOML text with a new nav list."""
    lines = toml_text.splitlines(keepends=True)
    start = end = None
    depth = 0
    for i, line in enumerate(lines):
        if start is None and re.match(r"^nav\s*=\s*\[", line):
            start = i
            depth = line.count("[") - line.count("]")
            if depth == 0:
                end = i
                break
        elif start is not None:
            depth += line.count("[") - line.count("]")
            if depth <= 0:
                end = i
                break
    if start is None:
        raise ValueError("nav key not found in zensical.toml")
    new_block = "nav = " + nav_to_toml_str(new_nav) + "\n"
    return "".join(lines[:start] + [new_block] + lines[end + 1:])


def patch_site_url(toml_text: str, new_url: str) -> str:
    """Replace site_url value in TOML text."""
    return re.sub(
        r'^(site_url\s*=\s*")([^"]*)(")',
        lambda m: f"{m.group(1)}{new_url}{m.group(3)}",
        toml_text,
        flags=re.MULTILINE,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    src_dir = repo_root / "src"
    toml_path = repo_root / "zensical.toml"
    out_path = repo_root / "_zensical_build.toml"

    toml_text = toml_path.read_text()
    with toml_path.open("rb") as f:
        config = tomllib.load(f)

    # Copy CONTRIBUTING.md so Zensical picks it up (symlinks and the name
    # "contributing.md" are both excluded by Zensical at build time)
    shutil.copy(repo_root / "CONTRIBUTING.md", docs_dir / "contributing.md")

    # Generate stubs and build API nav
    modules = scan_modules(src_dir)
    write_stubs(modules, docs_dir)
    api_nav = build_nav_tree(modules)
    print(f"Generated {len(modules)} API stubs")

    # Rebuild nav: keep everything except existing API entry, append new one
    base_nav = [item for item in config["project"]["nav"] if "API" not in item]
    new_nav = base_nav + [{"API": api_nav}]

    new_toml = replace_nav(toml_text, new_nav)

    # Optionally set versioned site_url
    alias = os.environ.get("VERSION_ALIAS", "").strip()
    if alias:
        base_url = config["project"]["site_url"].rstrip("/") + "/"
        new_toml = patch_site_url(new_toml, f"{base_url}{alias}/")

    out_path.write_text(new_toml)
    print(f"Written {out_path}")


if __name__ == "__main__":
    main()
