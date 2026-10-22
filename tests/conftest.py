"""Keep the test suite hermetic: no Supabase calls and no writes to project data."""

import atexit
import os
import shutil
import tempfile

os.environ["STORAGE_BACKEND"] = "local"
os.environ.pop("SUPABASE_URL", None)
os.environ.pop("SUPABASE_KEY", None)

_TEST_STATE_DIR = tempfile.mkdtemp(prefix="bookrag-test-state-")
os.environ["OBSERVABILITY_DIR"] = _TEST_STATE_DIR
atexit.register(shutil.rmtree, _TEST_STATE_DIR, True)
