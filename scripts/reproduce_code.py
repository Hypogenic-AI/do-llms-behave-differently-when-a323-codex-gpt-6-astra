"""Recreate exact resource snapshots without changing existing checkouts."""
from pathlib import Path
import json
import subprocess

root = Path.cwd()
print(root)
assert (root / ".git").is_dir(), "Run from workspace root"
for row in json.loads((root / "notes/code_manifest.json").read_text()):
    path = root / row["path"]
    if path.exists():
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != row["commit"]:
            raise RuntimeError(f"Existing checkout differs: {path}")
        continue
    path.mkdir(parents=True)
    subprocess.run(["git", "init", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "remote", "add", "origin", row["url"]], check=True)
    subprocess.run(["git", "-C", str(path), "fetch", "--depth=1", "origin", row["commit"]], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "--detach", row["commit"]], check=True)
print("All repository revisions verified.")
