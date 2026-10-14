from pathlib import Path
from urllib.request import Request, urlopen

BASE_URL = "https://raw.githubusercontent.com/Binyr/NormalCrafter/main/normalcrafter"
FILES = {
    "normal_crafter_ppl.py": f"{BASE_URL}/normal_crafter_ppl.py",
    "unet.py": f"{BASE_URL}/unet.py",
}

def ensure_upstream_code(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in FILES if not (target_dir / n).exists()]
    if not missing:
        return
    errors = []
    for name in missing:
        try:
            req = Request(FILES[name], headers={"User-Agent": "ComfyUI-VizClick-NormalCrafter"})
            with urlopen(req, timeout=30) as response:
                (target_dir / name).write_bytes(response.read())
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError(
            "Automatic download of the upstream NormalCrafter Python code failed. "
            "Internet access is required once on first use. "
            "Errors: " + "; ".join(errors)
        )
