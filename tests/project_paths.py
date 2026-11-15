"""Local test outputs and isolated comparison dependencies."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "build" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

def use_vsjet():
    """Use the isolated comparison package without changing installed packages."""
    import sys
    isolated = ROOT / '.tools/vsjet'
    if isolated.is_dir():
        sys.path.insert(0, str(isolated))
