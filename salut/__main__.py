"""Allow running the salut module directly: python -m salut [name]."""

import sys

from salut import salut


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else None
    print(salut(name))


if __name__ == "__main__":
    main()
