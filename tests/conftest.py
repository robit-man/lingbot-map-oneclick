from __future__ import annotations

import os


# webapp.app exposes the production ASGI object at import time. Keep test
# collection deterministic without reading the developer's private .env file.
os.environ.setdefault("LINGBOT_API_TOKEN", "test-token-that-is-at-least-24-characters")
