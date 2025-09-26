# In __init__.py
import sys

# Pyodide detection
is_pyodide = "pyodide" in sys.modules

# This is a simple way to "disable" gevent when running in Pyodide
# by replacing the module with a dummy object before it's imported elsewhere.
if is_pyodide:
    class DummyGevent:
        def sleep(self, seconds):
            # In a real async context, you would `await asyncio.sleep(seconds)`.
            # For now, this is a placeholder to prevent crashes.
            # The actual async logic will be in our new `main.py`.
            pass

    sys.modules["gevent"] = DummyGevent()