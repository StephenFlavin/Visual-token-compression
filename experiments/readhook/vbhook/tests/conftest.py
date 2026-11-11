import os
import pytest


@pytest.fixture(autouse=True)
def _dense_mode_for_family_tests(request, monkeypatch):
    """The family tests assert the dense variant (spaces between tokens removed); edit-safe mode is the default and is
    tested in test_edit_safe.py."""
    if os.path.basename(str(request.fspath)) in ("test_brace.py", "test_python_family.py"):
        monkeypatch.setenv("VBHOOK_TIGHT", "1")
