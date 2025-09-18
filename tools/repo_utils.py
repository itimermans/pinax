import subprocess
from pathlib import Path


def get_repo_root(start: Path | None = None) -> str:
    """
    Find the repository root directory.

    - First tries `git rev-parse --show-toplevel`
    - If git is not available or fails, walks up until it finds a `.git` folder.

    Parameters
    ----------
    start : Path | None
        Starting path (defaults to current working directory).

    Returns
    -------
    str
        Path to the repository root.

    Raises
    ------
    RuntimeError
        If no repository root is found.
    """
    start = start or Path.cwd()

    # --- Option 1: Try Git ---
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return str(Path(root))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # git failed or not installed

    # --- Option 2: Walk up until `.git` is found ---
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return str(parent)

    raise RuntimeError("Could not find repository root.")


if __name__ == "__main__":
    repo_root = get_repo_root()
    print("Repository root:", repo_root)
