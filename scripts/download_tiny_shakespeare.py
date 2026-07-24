from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "tinyshakespeare.txt"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=30) as response:  # noqa: S310 - explicit public dataset URL
        content = response.read()
    OUTPUT.write_bytes(content)
    print(f"Saved {len(content):,} bytes to {OUTPUT}")


if __name__ == "__main__":
    main()
