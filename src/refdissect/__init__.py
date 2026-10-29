"""reference-dissector: take a reference track apart (grid, tuning, drum hits,
808 notes, sounds) and rebuild it in Ableton Live through ableton-live-mcp."""

# first, before any other module of the package loads: the code this process runs (cache.LOADED)
from . import cache  # noqa: F401

__version__ = "0.1.0"
