"""Allow ``python -m src`` to invoke the codec CLI (delegates to src.codec)."""

import sys

from src.codec import main

if __name__ == "__main__":
    sys.exit(main())
