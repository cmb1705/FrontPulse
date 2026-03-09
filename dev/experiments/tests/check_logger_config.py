"""Diagnostic to check logger configuration."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.logging_config import setup_logging

# Setup the 2yp logger (simulating run.py)
logger_2yp = setup_logging(level="INFO", console=True, log_file=Path("data/out/logs/test_logger.log"))

# Create a logger like graph_build.py does
logger_graph_build = logging.getLogger("src.graph_build")

print("=" * 70)
print("LOGGER CONFIGURATION DIAGNOSTIC")
print("=" * 70)

# Check logger levels
print(f"\n2yp logger:")
print(f"  Name: {logger_2yp.name}")
print(f"  Level: {logging.getLevelName(logger_2yp.level)}")
print(f"  Handlers: {len(logger_2yp.handlers)}")
for i, handler in enumerate(logger_2yp.handlers):
    print(f"    Handler {i}: {type(handler).__name__}, level={logging.getLevelName(handler.level)}")

print(f"\nsrc.graph_build logger:")
print(f"  Name: {logger_graph_build.name}")
print(f"  Level: {logging.getLevelName(logger_graph_build.level)}")
print(f"  Effective level: {logging.getLevelName(logger_graph_build.getEffectiveLevel())}")
print(f"  Handlers: {len(logger_graph_build.handlers)}")
print(f"  Propagate: {logger_graph_build.propagate}")
for i, handler in enumerate(logger_graph_build.handlers):
    print(f"    Handler {i}: {type(handler).__name__}, level={logging.getLevelName(handler.level)}")

# Check parent logger
print(f"\nParent chain:")
current = logger_graph_build
depth = 0
while current:
    print(f"  {'  ' * depth}{current.name if current.name else 'root'} (level={logging.getLevelName(current.level)}, handlers={len(current.handlers)})")
    current = current.parent if hasattr(current, 'parent') else None
    depth += 1
    if depth > 5:  # Safety limit
        break

# Test logging
print(f"\n" + "=" * 70)
print("TESTING LOGGING OUTPUT")
print("=" * 70)

print("\nTesting 2yp logger:")
logger_2yp.info("TEST: This is from 2yp logger")
logger_2yp.warning("TEST: Warning from 2yp logger")

print("\nTesting src.graph_build logger:")
logger_graph_build.info("TEST: This is from src.graph_build logger")
logger_graph_build.warning("TEST: Warning from src.graph_build logger")

print("\nCheck data/out/logs/test_logger.log to see which messages appeared in the file.")
