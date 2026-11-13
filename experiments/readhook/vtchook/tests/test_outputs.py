# ruff: noqa: E501, W291, W293, E101  (the embedded samples are real output, trailing blanks and tabs included)
"""Command and tool output minifiers (minify/outputs.py): the base.py invariant on real command output, semantic
round trips (caret spans, diff line numbers, grep/find/status expansion), sniffing, command classification and edge
cases. No network, no Claude calls.

SAMPLES are real outputs (pytest 9 / Python 3.13 / gcc / clang / rustc / mypy / tsc / javac / ruff / eslint / go /
cargo / node --test / git / grep / rg / find / ls), captured on scratch projects and on this repository; scratch paths
are shortened to /home/dev/proj and long outputs are cut at a line boundary."""
import difflib
import json
import os
import re
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(HERE))
from vtchook.minify import REGISTRY, detect, minify, preserves  # noqa: E402
from vtchook.minify import outputs as O  # noqa: E402
from vtchook.minify.base import nonws  # noqa: E402

KINDS = ("pytest", "traceback", "diagnostics", "log", "gitdiff", "gitlog", "gitstatus", "grep", "find", "listing")

# fmt: off
SAMPLES = {
    'pytest_': ('pytest', '============================= test session starts ==============================\nplatform linux -- Python 3.13.18, pytest-9.2.0, pluggy-1.7.0\nrootdir: /home/dev/proj/pyt\nplugins: anyio-4.15.1\ncollected 21 items / 1 error\n\nsub/test_more.py FFF                                                     [ 14%]\ntest_calc.py F..F.FFFFFFFFFEsxF                                          [100%]\n\n==================================== ERRORS ====================================\n_____________________ ERROR collecting sub/test_broken.py ______________________\nImportError while importing test module \'/home/dev/proj/pyt/sub/test_broken.py\'.\nHint: make sure your test modules/packages have valid Python names.\nTraceback:\n/usr/local/lib/python3.13/importlib/__init__.py:88: in import_module\n    return _bootstrap._gcd_import(name[level:], package, level)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nsub/test_broken.py:1: in <module>\n    import numpy_not_installed_xyz\nE   ModuleNotFoundError: No module named \'numpy_not_installed_xyz\'\n________________________ ERROR at setup of test_with_db ________________________\n\n    @pytest.fixture\n    def db():\n>       raise RuntimeError("could not connect to database at localhost:5432")\nE       RuntimeError: could not connect to database at localhost:5432\n\ntest_calc.py:60: RuntimeError\n=================================== FAILURES ===================================\n_____________________________ TestPaths.test_join ______________________________\n\nself = <test_more.TestPaths object at 0x7f9521a5ccd0>\n\n    def test_join(self):\n>       assert os.path.join("a", "b") == "a\\\\b"\nE       AssertionError: assert \'a/b\' == \'a\\\\b\'\nE         \nE         - a\\b\nE         + a/b\n\nsub/test_more.py:6: AssertionError\n____________________________ TestPaths.test_exists _____________________________\n\nself = <test_more.TestPaths object at 0x7f9521a5d1d0>\ntmp_path = PosixPath(\'/tmp/pytest-of-root/pytest-115/test_exists0\')\n\n    def test_exists(self, tmp_path):\n        p = tmp_path / "missing.txt"\n>       assert p.exists(), f"expected {p} to exist"\nE       AssertionError: expected /tmp/pytest-of-root/pytest-115/test_exists0/missing.txt to exist\nE       assert False\nE        +  where False = exists()\nE        +    where exists = PosixPath(\'/tmp/pytest-of-root/pytest-115/test_exists0/missing.txt\').exists\n\nsub/test_more.py:10: AssertionError\n_____________________________ TestPaths.test_long ______________________________\n\nself = <test_more.TestPaths object at 0x7f95218b0180>\n\n    def test_long(self):\n        expected = list(range(50))\n        got = list(range(50))\n        got[37] = -1\n>       assert got == expected\nE       assert [0, 1, 2, 3, 4, 5, ...] == [0, 1, 2, 3, 4, 5, ...]\nE         \nE         At index 37 diff: -1 != 37\nE         Use -v to get more diff\n\nsub/test_more.py:16: AssertionError\n_________________________________ test_compute _________________________________\n\n    def test_compute():\n        x = compute(3)\n>       assert x == 10\nE       assert 9 == 10\n\ntest_calc.py:7: AssertionError\n___________________________ test_compute_param[2-7] ____________________________\n\nn = 2, expected = 7\n\n    @pytest.mark.parametrize("n,expected", [(1, 3), (2, 7), (4, 12), (5, 16)])\n    def test_compute_param(n, expected):\n>       assert compute(n) == expected\nE       assert 6 == 7\nE        +  where 6 = compute(2)\n\ntest_calc.py:16: AssertionError\n___________________________ test_compute_param[5-16] ___________________________\n\nn = 5, expected = 16\n\n    @pytest.mark.parametrize("n,expected", [(1, 3), (2, 7), (4, 12), (5, 16)])\n    def test_compute_param(n, expected):\n>       assert compute(n) == expected\nE       assert 15 == 16\nE        +  where 15 = compute(5)\n\ntest_calc.py:16: AssertionError\n_________________________________ test_strings _________________________________\n\n    def test_strings():\n>       assert "hello  world, this is a test" == "hello world, this is a test"\nE       AssertionError: assert \'hello  world, this is a test\' == \'hello world, this is a test\'\nE         \nE         - hello world, this is a test\nE         + hello  world, this is a test\nE         ?      +\n\ntest_calc.py:20: AssertionError\n__________________________________ test_lists __________________________________\n\n    def test_lists():\n>       assert [1, 2, 3, 4, 5] == [1, 2, 4, 4, 5]\nE       assert [1, 2, 3, 4, 5] == [1, 2, 4, 4, 5]\nE         \nE         At index 2 diff: 3 != 4\nE         Use -v to get more diff\n\ntest_calc.py:24: AssertionError\n__________________________________ test_dicts __________________________________\n\n    def test_dicts():\n        got = {"name": "server", "port": 8080, "debug": False, "hosts": ["a", "b"]}\n>       assert got == {"name": "server", "port": 8081, "debug": False, "hosts": ["a", "c"]}\nE       AssertionError: assert {\'name\': \'ser...\': [\'a\', \'b\']} == {\'name\': \'ser...\': [\'a\', \'c\']}\nE         \nE         Omitting 2 identical items, use -vv to show\nE         Differing items:\nE         {\'hosts\': [\'a\', \'b\']} != {\'hosts\': [\'a\', \'c\']}\nE         {\'port\': 8080} != {\'port\': 8081}\nE         Use -v to get more diff\n\ntest_calc.py:29: AssertionError\n________________________________ test_multiline ________________________________\n\n    def test_multiline():\n        a = "line one\\nline two\\nline three\\nline four"\n        b = "line one\\nline 2\\nline three\\nline four\\n"\n>       assert a == b\nE       AssertionError: assert \'line one\\nli...ee\\nline four\' == \'line one\\nli...\\nline four\\n\'\nE         \nE           line one\nE         - line 2\nE         ?      ^\nE         + line two\nE         ?      ^^^\nE           line three...\nE         \nE         ...Full output truncated, use \'-vv\' to show\n\ntest_calc.py:35: AssertionError\n__________________________________ test_parse __________________________________\n\n    def test_parse():\n>       assert parse_config(\'{"name": "x"}\') == ("x", 80)\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\ntest_calc.py:39: \n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n\ntext = \'{"name": "x"}\'\n\n    def parse_config(text):\n        data = json.loads(text)\n>       return data["name"], data["port"]\n                             ^^^^^^^^^^^^\nE       KeyError: \'port\'\n\ncalc.py:10: KeyError\n________________________________ test_pop_empty ________________________________\n\n    def test_pop_empty():\n        s = Stack()\n        s.push(1)\n        s.pop()\n>       s.pop()\n\ntest_calc.py:46: \n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n\nself = <calc.Stack object at 0x7f95218c0440>\n\n    def pop(self):\n>       return self.items.pop()\n               ^^^^^^^^^^^^^^^^\nE       IndexError: pop from empty list\n\ncalc.py:21: IndexError\n__________________________________ test_ratio __________________________________\n\n    def test_ratio():\n>       assert ratio((1, 2), (3, 0)) == 1\n               ^^^^^^^^^^^^^^^^^^^^^\n\ntest_calc.py:50: \n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n\na = (1, 2), b = (3, 0)\n\n    def ratio(a, b):\n>       return a[0] / b[1] + a[1]\n               ^^^^^^^^^^^\nE       ZeroDivisionError: division by zero\n\ncalc.py:25: ZeroDivisionError\n_________________________________ test_raises __________________________________\n\n    def test_raises():\n>       with pytest.raises(ValueError):\n             ^^^^^^^^^^^^^^^^^^^^^^^^^\nE       Failed: DID NOT RAISE ValueError\n\ntest_calc.py:54: Failed\n_________________________________ test_output __________________________________\n\ncapsys = <_pytest.capture.CaptureFixture object at 0x7f95218c0440>\n\n    def test_output(capsys):\n        print("debug: starting")\n        print("debug: value =", compute(7))\n>       assert compute(7) == 20\nE       assert 21 == 20\nE        +  where 21 = compute(7)\n\ntest_calc.py:80: AssertionError\n----------------------------- Captured stdout call -----------------------------\ndebug: starting\ndebug: value = 21\n=========================== short test summary info ============================\nFAILED sub/test_more.py::TestPaths::test_join - AssertionError: assert \'a/b\' ...\nFAILED sub/test_more.py::TestPaths::test_exists - AssertionError: expected /t...\nFAILED sub/test_more.py::TestPaths::test_long - assert [0, 1, 2, 3, 4, 5, ......\nFAILED test_calc.py::test_compute - assert 9 == 10\nFAILED test_calc.py::test_compute_param[2-7] - assert 6 == 7\nFAILED test_calc.py::test_compute_param[5-16] - assert 15 == 16\nFAILED test_calc.py::test_strings - AssertionError: assert \'hello  world, thi...\nFAILED test_calc.py::test_lists - assert [1, 2, 3, 4, 5] == [1, 2, 4, 4, 5]\nFAILED test_calc.py::test_dicts - AssertionError: assert {\'name\': \'ser...\': [...\nFAILED test_calc.py::test_multiline - AssertionError: assert \'line one\\nli......\nFAILED test_calc.py::test_parse - KeyError: \'port\'\nFAILED test_calc.py::test_pop_empty - IndexError: pop from empty list\nFAILED test_calc.py::test_ratio - ZeroDivisionError: division by zero\nFAILED test_calc.py::test_raises - Failed: DID NOT RAISE ValueError\n'),
    'pytest_-v': ('pytest', r"""============================= test session starts ==============================
platform linux -- Python 3.13.18, pytest-9.2.0, pluggy-1.7.0 -- /usr/local/bin/python3
rootdir: /home/dev/proj/pyt
plugins: anyio-4.15.1
collecting ... collected 21 items / 1 error

sub/test_more.py::TestPaths::test_join FAILED                            [  4%]
sub/test_more.py::TestPaths::test_exists FAILED                          [  9%]
sub/test_more.py::TestPaths::test_long FAILED                            [ 14%]
test_calc.py::test_compute FAILED                                        [ 19%]
test_calc.py::test_compute_ok PASSED                                     [ 23%]
test_calc.py::test_compute_param[1-3] PASSED                             [ 28%]
test_calc.py::test_compute_param[2-7] FAILED                             [ 33%]
test_calc.py::test_compute_param[4-12] PASSED                            [ 38%]
test_calc.py::test_compute_param[5-16] FAILED                            [ 42%]
test_calc.py::test_strings FAILED                                        [ 47%]
test_calc.py::test_lists FAILED                                          [ 52%]
test_calc.py::test_dicts FAILED                                          [ 57%]
test_calc.py::test_multiline FAILED                                      [ 61%]
test_calc.py::test_parse FAILED                                          [ 66%]
test_calc.py::test_pop_empty FAILED                                      [ 71%]
test_calc.py::test_ratio FAILED                                          [ 76%]
test_calc.py::test_raises FAILED                                         [ 80%]
test_calc.py::test_with_db ERROR                                         [ 85%]
test_calc.py::test_skipped SKIPPED (not ready)                           [ 90%]
test_calc.py::test_xfail XFAIL (known bug)                               [ 95%]
test_calc.py::test_output FAILED                                         [100%]

==================================== ERRORS ====================================
_____________________ ERROR collecting sub/test_broken.py ______________________
ImportError while importing test module '/home/dev/proj/pyt/sub/test_broken.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/local/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
sub/test_broken.py:1: in <module>
    import numpy_not_installed_xyz
E   ModuleNotFoundError: No module named 'numpy_not_installed_xyz'
"""),
    'pytest_--tb_short': ('pytest', '============================= test session starts ==============================\nplatform linux -- Python 3.13.18, pytest-9.2.0, pluggy-1.7.0\nrootdir: /home/dev/proj/pyt\nplugins: anyio-4.15.1\ncollected 21 items / 1 error\n\nsub/test_more.py FFF                                                     [ 14%]\ntest_calc.py F..F.FFFFFFFFFEsxF                                          [100%]\n\n==================================== ERRORS ====================================\n_____________________ ERROR collecting sub/test_broken.py ______________________\nImportError while importing test module \'/home/dev/proj/pyt/sub/test_broken.py\'.\nHint: make sure your test modules/packages have valid Python names.\nTraceback:\n/usr/local/lib/python3.13/importlib/__init__.py:88: in import_module\n    return _bootstrap._gcd_import(name[level:], package, level)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nsub/test_broken.py:1: in <module>\n    import numpy_not_installed_xyz\nE   ModuleNotFoundError: No module named \'numpy_not_installed_xyz\'\n________________________ ERROR at setup of test_with_db ________________________\ntest_calc.py:60: in db\n    raise RuntimeError("could not connect to database at localhost:5432")\nE   RuntimeError: could not connect to database at localhost:5432\n=================================== FAILURES ===================================\n_____________________________ TestPaths.test_join ______________________________\nsub/test_more.py:6: in test_join\n    assert os.path.join("a", "b") == "a\\\\b"\nE   AssertionError: assert \'a/b\' == \'a\\\\b\'\nE     \nE     - a\\b\nE     + a/b\n____________________________ TestPaths.test_exists _____________________________\nsub/test_more.py:10: in test_exists\n    assert p.exists(), f"expected {p} to exist"\nE   AssertionError: expected /tmp/pytest-of-root/pytest-118/test_exists0/missing.txt to exist\nE   assert False\nE    +  where False = exists()\nE    +    where exists = PosixPath(\'/tmp/pytest-of-root/pytest-118/test_exists0/missing.txt\').exists\n_____________________________ TestPaths.test_long ______________________________\nsub/test_more.py:16: in test_long\n    assert got == expected\nE   assert [0, 1, 2, 3, 4, 5, ...] == [0, 1, 2, 3, 4, 5, ...]\nE     \nE     At index 37 diff: -1 != 37\nE     Use -v to get more diff\n_________________________________ test_compute _________________________________\ntest_calc.py:7: in test_compute\n    assert x == 10\nE   assert 9 == 10\n___________________________ test_compute_param[2-7] ____________________________\ntest_calc.py:16: in test_compute_param\n    assert compute(n) == expected\nE   assert 6 == 7\n'),
    'pytest_--tb_native': ('pytest', r"""============================= test session starts ==============================
platform linux -- Python 3.13.18, pytest-9.2.0, pluggy-1.7.0
rootdir: /home/dev/proj/pyt
plugins: anyio-4.15.1
collected 21 items / 1 error

sub/test_more.py FFF                                                     [ 14%]
test_calc.py F..F.FFFFFFFFFEsxF                                          [100%]

==================================== ERRORS ====================================
_____________________ ERROR collecting sub/test_broken.py ______________________
ImportError while importing test module '/home/dev/proj/pyt/sub/test_broken.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/local/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
sub/test_broken.py:1: in <module>
    import numpy_not_installed_xyz
E   ModuleNotFoundError: No module named 'numpy_not_installed_xyz'
________________________ ERROR at setup of test_with_db ________________________
Traceback (most recent call last):
  File "/usr/local/lib/python3.13/site-packages/_pytest/runner.py", line 361, in from_call
    result: TResult | None = func()
                             ~~~~^^
  File "/usr/local/lib/python3.13/site-packages/_pytest/runner.py", line 250, in <lambda>
    lambda: runtest_hook(item=item, **kwds),
            ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/pluggy/_caller.py", line 197, in __call__
    return self._hookexec(
           ~~~~~~~~~~~~~~^
        self.name, self._hookimpls.copy(), call_kwargs, firstresult
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.13/site-packages/pluggy/_manager.py", line 183, in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/pluggy/_execution.py", line 195, in _multicall
    raise exception
  File "/usr/local/lib/python3.13/site-packages/pluggy/_execution.py", line 167, in _multicall
    teardown.throw(exception)
    ~~~~~~~~~~~~~~^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/_pytest/logging.py", line 871, in pytest_runtest_setup
    yield
  File "/usr/local/lib/python3.13/site-packages/pluggy/_execution.py", line 167, in _multicall
    teardown.throw(exception)
    ~~~~~~~~~~~~~~^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/_pytest/capture.py", line 901, in pytest_runtest_setup
    return (yield)
            ^^^^^
  File "/usr/local/lib/python3.13/site-packages/pluggy/_execution.py", line 147, in _multicall
    res = hook_impl.function(*args)
  File "/usr/local/lib/python3.13/site-packages/_pytest/runner.py", line 170, in pytest_runtest_setup
    item.session._setupstate.setup(item)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/usr/local/lib/python3.13/site-packages/_pytest/runner.py", line 536, in setup
    col.setup()
    ~~~~~~~~~^^
"""),
    'pytest_color': ('pytest', '\x1b[31mF\x1b[0m\x1b[31mF\x1b[0m\x1b[31mF\x1b[0m\x1b[31mF\x1b[0m\x1b[32m.\x1b[0m\x1b[32m.\x1b[0m\x1b[31mF\x1b[0m\x1b[32m.\x1b[0m\x1b[31mF\x1b[0m\x1b[31mF\x1b[0m\x1b[31mF\x1b[0m\x1b[31mF\x1b[0m\x1b[31mF\x1b[0m\x1b[31mF\x1b[0m\x1b[31mF\x1b[0m\x1b[31mF\x1b[0m\x1b[31mF\x1b[0m\x1b[31mE\x1b[0m\x1b[33ms\x1b[0m\x1b[33mx\x1b[0m\x1b[31mF\x1b[0m\x1b[31m                                                    [100%]\x1b[0m\n==================================== ERRORS ====================================\n\x1b[31m\x1b[1m_____________________ ERROR collecting sub/test_broken.py ______________________\x1b[0m\n\x1b[31mImportError while importing test module \'/home/dev/proj/pyt/sub/test_broken.py\'.\nHint: make sure your test modules/packages have valid Python names.\nTraceback:\n/usr/local/lib/python3.13/importlib/__init__.py:88: in import_module\n    return _bootstrap._gcd_import(name[level:], package, level)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nsub/test_broken.py:1: in <module>\n    import numpy_not_installed_xyz\nE   ModuleNotFoundError: No module named \'numpy_not_installed_xyz\'\x1b[0m\n\x1b[31m\x1b[1m________________________ ERROR at setup of test_with_db ________________________\x1b[0m\n\n    \x1b[0m\x1b[37m@pytest\x1b[39;49;00m.fixture\x1b[90m\x1b[39;49;00m\n    \x1b[94mdef\x1b[39;49;00m\x1b[90m \x1b[39;49;00m\x1b[92mdb\x1b[39;49;00m():\x1b[90m\x1b[39;49;00m\n>       \x1b[94mraise\x1b[39;49;00m \x1b[96mRuntimeError\x1b[39;49;00m(\x1b[33m"\x1b[39;49;00m\x1b[33mcould not connect to database at localhost:5432\x1b[39;49;00m\x1b[33m"\x1b[39;49;00m)\x1b[90m\x1b[39;49;00m\n\x1b[1m\x1b[31mE       RuntimeError: could not connect to database at localhost:5432\x1b[0m\n\n\x1b[1m\x1b[31mtest_calc.py\x1b[0m:60: RuntimeError\n=================================== FAILURES ===================================\n\x1b[31m\x1b[1m_____________________________ TestPaths.test_join ______________________________\x1b[0m\n\nself = <test_more.TestPaths object at 0x7faa2ca8b250>\n\n    \x1b[0m\x1b[94mdef\x1b[39;49;00m\x1b[90m \x1b[39;49;00m\x1b[92mtest_join\x1b[39;49;00m(\x1b[96mself\x1b[39;49;00m):\x1b[90m\x1b[39;49;00m\n>       \x1b[94massert\x1b[39;49;00m os.path.join(\x1b[33m"\x1b[39;49;00m\x1b[33ma\x1b[39;49;00m\x1b[33m"\x1b[39;49;00m, \x1b[33m"\x1b[39;49;00m\x1b[33mb\x1b[39;49;00m\x1b[33m"\x1b[39;49;00m) == \x1b[33m"\x1b[39;49;00m\x1b[33ma\x1b[39;49;00m\x1b[33m\\\\\x1b[39;49;00m\x1b[33mb\x1b[39;49;00m\x1b[33m"\x1b[39;49;00m\x1b[90m\x1b[39;49;00m\n\x1b[1m\x1b[31mE       AssertionError: assert \'a/b\' == \'a\\\\b\'\x1b[0m\n\x1b[1m\x1b[31mE         \x1b[0m\n\x1b[1m\x1b[31mE         \x1b[0m\x1b[91m- a\\b\x1b[39;49;00m\x1b[90m\x1b[39;49;00m\x1b[0m\n\x1b[1m\x1b[31mE         \x1b[92m+ a/b\x1b[39;49;00m\x1b[90m\x1b[39;49;00m\x1b[0m\n\n\x1b[1m\x1b[31msub/test_more.py\x1b[0m:6: AssertionError\n\x1b[31m\x1b[1m____________________________ TestPaths.test_exists _____________________________\x1b[0m\n\nself = <test_more.TestPaths object at 0x7faa2ca8b750>\ntmp_path = PosixPath(\'/tmp/pytest-of-root/pytest-123/test_exists0\')\n\n    \x1b[0m\x1b[94mdef\x1b[39;49;00m\x1b[90m \x1b[39;49;00m\x1b[92mtest_exists\x1b[39;49;00m(\x1b[96mself\x1b[39;49;00m, tmp_path):\x1b[90m\x1b[39;49;00m\n        p = tmp_path / \x1b[33m"\x1b[39;49;00m\x1b[33mmissing.txt\x1b[39;49;00m\x1b[33m"\x1b[39;49;00m\x1b[90m\x1b[39;49;00m\n>       \x1b[94massert\x1b[39;49;00m p.exists(), \x1b[33mf\x1b[39;49;00m\x1b[33m"\x1b[39;49;00m\x1b[33mexpected \x1b[39;49;00m\x1b[33m{\x1b[39;49;00mp\x1b[33m}\x1b[39;49;00m\x1b[33m to exist\x1b[39;49;00m\x1b[33m"\x1b[39;49;00m\x1b[90m\x1b[39;49;00m\n\x1b[1m\x1b[31mE       AssertionError: expected /tmp/pytest-of-root/pytest-123/test_exists0/missing.txt to exist\x1b[0m\n\x1b[1m\x1b[31mE       assert False\x1b[0m\n\x1b[1m\x1b[31mE        +  where False = exists()\x1b[0m\n\x1b[1m\x1b[31mE        +    where exists = PosixPath(\'/tmp/pytest-of-root/pytest-123/test_exists0/missing.txt\').exists\x1b[0m\n\n\x1b[1m\x1b[31msub/test_more.py\x1b[0m:10: AssertionError\n\x1b[31m\x1b[1m_____________________________ TestPaths.test_long ______________________________\x1b[0m\n\nself = <test_more.TestPaths object at 0x7faa2ca7c180>\n\n'),
    'pytest_collect_error': ('pytest', r"""============================= test session starts ==============================
platform linux -- Python 3.13.18, pytest-9.2.0, pluggy-1.7.0
rootdir: /home/dev/proj/pyt
plugins: anyio-4.15.1
collected 21 items / 1 error

==================================== ERRORS ====================================
_____________________ ERROR collecting sub/test_broken.py ______________________
ImportError while importing test module '/home/dev/proj/pyt/sub/test_broken.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/local/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
sub/test_broken.py:1: in <module>
    import numpy_not_installed_xyz
E   ModuleNotFoundError: No module named 'numpy_not_installed_xyz'
=========================== short test summary info ============================
ERROR sub/test_broken.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.08s ===============================
"""),
    'pytb_chain': ('traceback', r"""Traceback (most recent call last):
  File "/home/dev/proj/tb/app.py", line 28, in main
    cfg.load()
    ~~~~~~~~^^
  File "/home/dev/proj/tb/app.py", line 10, in load
    with open(self.path) as f:
         ~~~~^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'missing.json'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/dev/proj/tb/app.py", line 37, in <module>
    main()
    ~~~~^^
  File "/home/dev/proj/tb/app.py", line 30, in main
    raise RuntimeError(f"cannot load config {cfg.path}") from e
RuntimeError: cannot load config missing.json
"""),
    'pytb_zero': ('traceback', r"""Traceback (most recent call last):
  File "/home/dev/proj/tb/app.py", line 39, in <module>
    print(process([{"value": 3, "weight": 2}, {"value": 1, "weight": 0}], 2.5))
          ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dev/proj/tb/app.py", line 21, in process
    total += item["value"] * scale / item["weight"]
             ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
"""),
    'pytb_group': ('traceback', r"""  + Exception Group Traceback (most recent call last):
  |   File "/home/dev/proj/tb/app.py", line 53, in <module>
  |     raise ExceptionGroup("many errors", [ValueError("bad value"), TypeError("bad type")])
  | ExceptionGroup: many errors (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | ValueError: bad value
    +---------------- 2 ----------------
    | TypeError: bad type
    +------------------------------------
"""),
    'pytb_syntax': ('traceback', r"""  File "/home/dev/proj/tb/bad.py", line 1
    def f(:
          ^
SyntaxError: invalid syntax
"""),
    'pytb_tab': ('traceback', r"""  File "/home/dev/proj/tb/bad3.py", line 3
    	pass
    ^
TabError: inconsistent use of tabs and spaces in indentation
"""),
    'pytb_deep': ('traceback', r"""Traceback (most recent call last):
  File "/home/dev/proj/tb/deep.py", line 13, in run
    save("/tmp/out.json", [Item("a", {"x", "y"})])
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dev/proj/tb/deep.py", line 10, in save
    f.write(encode(items))
            ~~~~~~^^^^^^^
  File "/home/dev/proj/tb/deep.py", line 7, in encode
    return json.dumps([dataclasses.asdict(i) for i in items], indent=2)
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/json/__init__.py", line 242, in dumps
    **kw).encode(obj)
          ~~~~~~^^^^^
  File "/usr/local/lib/python3.13/json/encoder.py", line 202, in encode
    chunks = self.iterencode(o, _one_shot=True)
  File "/usr/local/lib/python3.13/json/encoder.py", line 263, in iterencode
    return _iterencode(o, 0)
  File "/usr/local/lib/python3.13/json/encoder.py", line 182, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
                    f'is not JSON serializable')
TypeError: Object of type set is not JSON serializable

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/dev/proj/tb/deep.py", line 16, in <module>
    run()
    ~~~^^
  File "/home/dev/proj/tb/deep.py", line 15, in run
    raise ValueError("could not save items") from e
ValueError: could not save items
"""),
    'pytb_recursion': ('traceback', r"""Traceback (most recent call last):
  File "/home/dev/proj/tb/app.py", line 46, in <module>
    f(0)
    ~^^^
  File "/home/dev/proj/tb/app.py", line 45, in f
    return f(n + 1)
  File "/home/dev/proj/tb/app.py", line 45, in f
    return f(n + 1)
  File "/home/dev/proj/tb/app.py", line 45, in f
    return f(n + 1)
  [Previous line repeated 996 more times]
RecursionError: maximum recursion depth exceeded
"""),
    'pytb_torch': ('traceback', r"""Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch; torch.zeros(3,4) @ torch.zeros(5,6)
                  ~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~
RuntimeError: mm: mat1 and mat2 shapes cannot be multiplied (3x4 and 5x6)
"""),
    'pytb_subprocess': ('traceback', r"""Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import subprocess; subprocess.run(['definitely-not-a-command', '--x'], check=True)
                       ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/subprocess.py", line 555, in run
    with Popen(*popenargs, **kwargs) as process:
         ~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/subprocess.py", line 1040, in __init__
    self._execute_child(args, executable, preexec_fn, close_fds,
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        pass_fds, cwd, env,
                        ^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
                        gid, gids, uid, umask,
                        ^^^^^^^^^^^^^^^^^^^^^^
                        start_new_session, process_group)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/subprocess.py", line 1992, in _execute_child
    raise child_exception_type(errno_num, err_msg, err_filename)
FileNotFoundError: [Errno 2] No such file or directory: 'definitely-not-a-command'
"""),
    'diag_gcc': ('diagnostics', 'main.c: In function \'area\':\nmain.c:7:31: error: expected \';\' before \'}\' token\n    7 |     return p->x * p->y * scale\n      |                               ^\n      |                               ;\n    8 | }\n      | ~                              \nmain.c: In function \'main\':\nmain.c:12:13: error: \'undeclared_var\' undeclared (first use in this function)\n   12 |     int n = undeclared_var + 1;\n      |             ^~~~~~~~~~~~~~\nmain.c:12:13: note: each undeclared identifier is reported only once for each function it appears in\nmain.c:15:14: warning: format \'%s\' expects argument of type \'char *\', but argument 2 has type \'int\' [-Wformat=]\n   15 |     printf("%s %d\\n", area(&p, 2));\n      |             ~^        ~~~~~~~~~~~\n      |              |        |\n      |              char *   int\n      |             %d\nmain.c:15:17: warning: format \'%d\' expects a matching \'int\' argument [-Wformat=]\n   15 |     printf("%s %d\\n", area(&p, 2));\n      |                ~^\n      |                 |\n      |                 int\nmain.c:17:11: warning: comparison of unsigned expression in \'< 0\' is always false [-Wtype-limits]\n   17 |     if (u < 0) return 1;\n      |           ^\nmain.c:18:15: error: incompatible types when initializing type \'float\' using type \'char *\'\n   18 |     float f = "text";\n      |               ^~~~~~\nmain.c:19:17: error: incompatible type for argument 1 of \'area\'\n   19 |     return area(p, 2);\n      |                 ^\n      |                 |\n      |                 struct point\nmain.c:6:31: note: expected \'struct point *\' but argument is of type \'struct point\'\n    6 | static int area(struct point *p, int scale) {\n      |                 ~~~~~~~~~~~~~~^\nmain.c:18:11: warning: unused variable \'f\' [-Wunused-variable]\n   18 |     float f = "text";\n      |           ^\nmain.c:12:9: warning: unused variable \'n\' [-Wunused-variable]\n   12 |     int n = undeclared_var + 1;\n      |         ^\nmain.c:10:14: warning: unused parameter \'argc\' [-Wunused-parameter]\n   10 | int main(int argc, char **argv) {\n      |          ~~~~^~~~\n'),
    'diag_clang': ('diagnostics', 'main.c:7:31: error: expected \';\' after return statement\n    7 |     return p->x * p->y * scale\n      |                               ^\n      |                               ;\nmain.c:12:13: error: use of undeclared identifier \'undeclared_var\'\n   12 |     int n = undeclared_var + 1;\n      |             ^\nmain.c:15:23: warning: format specifies type \'char *\' but the argument has type \'int\' [-Wformat]\n   15 |     printf("%s %d\\n", area(&p, 2));\n      |             ~~        ^~~~~~~~~~~\n      |             %d\nmain.c:15:17: warning: more \'%\' conversions than data arguments [-Wformat-insufficient-args]\n   15 |     printf("%s %d\\n", area(&p, 2));\n      |                ~^\nmain.c:18:11: error: initializing \'float\' with an expression of incompatible type \'char[5]\'\n   18 |     float f = "text";\n      |           ^   ~~~~~~\nmain.c:19:17: error: passing \'struct point\' to parameter of incompatible type \'struct point *\'; take the address with &\n   19 |     return area(p, 2);\n      |                 ^\n      |                 &\nmain.c:6:31: note: passing argument to parameter \'p\' here\n    6 | static int area(struct point *p, int scale) {\n      |                               ^\nmain.c:10:14: warning: unused parameter \'argc\' [-Wunused-parameter]\n   10 | int main(int argc, char **argv) {\n      |              ^\n3 warnings and 4 errors generated.\n'),
    'diag_gpp': ('diagnostics', r"""util.cpp: In function 'int main()':
util.cpp:8:13: error: no matching function for call to 'std::map<std::__cxx11::basic_string<char>, int>::insert(std::vector<std::__cxx11::basic_string<char> >&)'
    8 |     m.insert(names);
      |     ~~~~~~~~^~~~~~~
In file included from /usr/include/c++/13/map:63,
                 from util.cpp:3:
/usr/include/c++/13/bits/stl_map.h:848:9: note: candidate: 'template<class _Pair> std::__enable_if_t<std::is_constructible<std::pair<const _Key, _Tp>, _Pair>::value, std::pair<typename std::_Rb_tree<_Key, std::pair<const _Key, _Tp>, std::_Select1st<std::pair<const _Key, _Tp> >, _Compare, typename __gnu_cxx::__alloc_traits<_Allocator>::rebind<std::pair<const _Key, _Tp> >::other>::iterator, bool> > std::map<_Key, _Tp, _Compare, _Alloc>::insert(_Pair&&) [with _Key = std::__cxx11::basic_string<char>; _Tp = int; _Compare = std::less<std::__cxx11::basic_string<char> >; _Alloc = std::allocator<std::pair<const std::__cxx11::basic_string<char>, int> >]'
  848 |         insert(_Pair&& __x)
      |         ^~~~~~
/usr/include/c++/13/bits/stl_map.h:848:9: note:   template argument deduction/substitution failed:
In file included from /usr/include/c++/13/bits/stl_pair.h:60,
                 from /usr/include/c++/13/bits/stl_algobase.h:64,
                 from /usr/include/c++/13/vector:62,
                 from util.cpp:1:
/usr/include/c++/13/type_traits: In substitution of 'template<bool _Cond, class _Tp> using std::__enable_if_t = typename std::enable_if::type [with bool _Cond = false; _Tp = std::pair<std::_Rb_tree_iterator<std::pair<const std::__cxx11::basic_string<char>, int> >, bool>]':
/usr/include/c++/13/bits/stl_map.h:848:2:   required by substitution of 'template<class _Pair> std::__enable_if_t<std::is_constructible<std::pair<const std::__cxx11::basic_string<char>, int>, _Pair>::value, std::pair<std::_Rb_tree_iterator<std::pair<const std::__cxx11::basic_string<char>, int> >, bool> > std::map<std::__cxx11::basic_string<char>, int>::insert(_Pair&&) [with _Pair = std::vector<std::__cxx11::basic_string<char> >&]'
util.cpp:8:13:   required from here
/usr/include/c++/13/type_traits:116:11: error: no type named 'type' in 'struct std::enable_if<false, std::pair<std::_Rb_tree_iterator<std::pair<const std::__cxx11::basic_string<char>, int> >, bool> >'
  116 |     using __enable_if_t = typename enable_if<_Cond, _Tp>::type;
      |           ^~~~~~~~~~~~~
"""),
    'diag_cargo_build': ('diagnostics', r"""   Compiling rs v0.1.0 (/home/dev/proj/rs)
error[E0308]: mismatched types
  --> src/main.rs:14:18
   |
14 |     let x: i32 = "a";
   |            ---   ^^^ expected `i32`, found `&str`
   |            |
   |            expected due to this

error[E0308]: mismatched types
  --> src/main.rs:18:19
   |
18 |     let s = total(v);
   |             ----- ^ expected `&Vec<i32>`, found `Vec<{integer}>`
   |             |
   |             arguments to this function are incorrect
   |
   = note: expected reference `&Vec<i32>`
                 found struct `Vec<{integer}>`
note: function defined here
  --> src/main.rs:7:4
   |
 7 | fn total(v: &Vec<i32>) -> i32 {
   |    ^^^^^ ------------
help: consider borrowing here
   |
18 |     let s = total(&v);
   |                   +

error[E0277]: cannot add `Option<&i32>` to `i32`
  --> src/main.rs:19:22
   |
19 |     println!("{}", s + m.get("k"));
   |                      ^ no implementation for `i32 + Option<&i32>`
   |
   = help: the trait `Add<Option<&i32>>` is not implemented for `i32`
help: the following other types implement trait `Add<Rhs>`
  --> /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/core/src/ops/arith.rs:98:8
   |
   = note: `i32` implements `Add`
  ::: /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/core/src/ops/arith.rs:113:0
   |
   = note: in this macro invocation
  --> /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/core/src/internal_macros.rs:22:8
   |
   = note: `&i32` implements `Add<i32>`
  ::: /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/core/src/internal_macros.rs:33:8
   |
   = note: `i32` implements `Add<&i32>`
  ::: /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/core/src/internal_macros.rs:44:8
   |
   = note: `&i32` implements `Add`
   = note: this error originates in the macro `add_impl` (in Nightly builds, run with -Z macro-backtrace for more info)

Some errors have detailed explanations: E0277, E0308.
For more information about an error, try `rustc --explain E0277`.
error: could not compile `rs` (bin "rs") due to 3 previous errors
"""),
    'diag_mypy': ('diagnostics', r"""lintme.py:11: error: Unsupported operand types for + ("str" and "int")  [operator]
lintme.py:19: error: Incompatible return value type (got "int", expected "str")  [return-value]
lintme.py:22: error: Argument 1 to "add" has incompatible type "str"; expected "int"  [arg-type]
lintme.py:23: error: "Box" has no attribute "frist"; maybe "first"?  [attr-defined]
lintme.py:24: error: Name "undefined_name" is not defined  [name-defined]
Found 5 errors in 1 file (checked 1 source file)
"""),
    'diag_mypy_pretty': ('diagnostics', r"""lintme.py:11: error: Unsupported operand types for + ("str" and "int") 
[operator]
        return "hi " + name + 3
                              ^
lintme.py:19: error: Incompatible return value type (got "int", expected "str")
 [return-value]
            return self.items[0]
                   ^~~~~~~~~~~~~
lintme.py:22: error: Argument 1 to "add" has incompatible type "str"; expected
"int"  [arg-type]
    add("1", 2)
        ^~~
lintme.py:23: error: "Box" has no attribute "frist"; maybe "first"? 
[attr-defined]
    Box([1]).frist()
    ^~~~~~~~~~~~~~
lintme.py:24: error: Name "undefined_name" is not defined  [name-defined]
    print(undefined_name)
          ^~~~~~~~~~~~~~
Found 5 errors in 1 file (checked 1 source file)
"""),
    'diag_tsc': ('diagnostics', r"""src/app.ts(4,23): error TS2551: Property 'nme' does not exist on type 'User'. Did you mean 'name'?
src/app.ts(7,26): error TS2322: Type 'string' is not assignable to type 'number'.
src/app.ts(8,5): error TS2322: Type 'string' is not assignable to type 'number'.
src/app.ts(10,6): error TS2322: Type 'string' is not assignable to type 'number'.
src/app.ts(10,11): error TS2322: Type 'string' is not assignable to type 'number'.
src/app.ts(11,7): error TS2345: Argument of type '{ id: number; }' is not assignable to parameter of type 'User'.
  Property 'name' is missing in type '{ id: number; }' but required in type 'User'.
src/app.ts(13,1): error TS18047: 'el' is possibly 'null'.
src/app.ts(13,1): error TS2322: Type 'number' is not assignable to type 'string'.
"""),
    'diag_tsc_pretty': ('diagnostics', '\x1b[96msrc/app.ts\x1b[0m:\x1b[93m4\x1b[0m:\x1b[93m23\x1b[0m - \x1b[91merror\x1b[0m\x1b[90m TS2551: \x1b[0mProperty \'nme\' does not exist on type \'User\'. Did you mean \'name\'?\n\n\x1b[7m4\x1b[0m   return "Hello " + u.nme;\n\x1b[7m \x1b[0m \x1b[91m                      ~~~\x1b[0m\n\n  \x1b[96msrc/app.ts\x1b[0m:\x1b[93m1\x1b[0m:\x1b[93m30\x1b[0m\n    \x1b[7m1\x1b[0m interface User { id: number; name: string; email?: string }\n    \x1b[7m \x1b[0m \x1b[96m                             ~~~~\x1b[0m\n    \'name\' is declared here.\n\n\x1b[96msrc/app.ts\x1b[0m:\x1b[93m7\x1b[0m:\x1b[93m26\x1b[0m - \x1b[91merror\x1b[0m\x1b[90m TS2322: \x1b[0mType \'string\' is not assignable to type \'number\'.\n\n\x1b[7m7\x1b[0m const users: User[] = [{ id: "1", name: "a" }];\n\x1b[7m \x1b[0m \x1b[91m                         ~~\x1b[0m\n\n  \x1b[96msrc/app.ts\x1b[0m:\x1b[93m1\x1b[0m:\x1b[93m18\x1b[0m\n    \x1b[7m1\x1b[0m interface User { id: number; name: string; email?: string }\n    \x1b[7m \x1b[0m \x1b[96m                 ~~\x1b[0m\n    The expected type comes from property \'id\' which is declared here on type \'User\'\n\n\x1b[96msrc/app.ts\x1b[0m:\x1b[93m8\x1b[0m:\x1b[93m5\x1b[0m - \x1b[91merror\x1b[0m\x1b[90m TS2322: \x1b[0mType \'string\' is not assignable to type \'number\'.\n\n\x1b[7m8\x1b[0m let count: number = "zero";\n\x1b[7m \x1b[0m \x1b[91m    ~~~~~\x1b[0m\n\n\x1b[96msrc/app.ts\x1b[0m:\x1b[93m10\x1b[0m:\x1b[93m6\x1b[0m - \x1b[91merror\x1b[0m\x1b[90m TS2322: \x1b[0mType \'string\' is not assignable to type \'number\'.\n\n\x1b[7m10\x1b[0m sum(["1", "2"]);\n\x1b[7m  \x1b[0m \x1b[91m     ~~~\x1b[0m\n\n\x1b[96msrc/app.ts\x1b[0m:\x1b[93m10\x1b[0m:\x1b[93m11\x1b[0m - \x1b[91merror\x1b[0m\x1b[90m TS2322: \x1b[0mType \'string\' is not assignable to type \'number\'.\n\n\x1b[7m10\x1b[0m sum(["1", "2"]);\n\x1b[7m  \x1b[0m \x1b[91m          ~~~\x1b[0m\n\n\x1b[96msrc/app.ts\x1b[0m:\x1b[93m11\x1b[0m:\x1b[93m7\x1b[0m - \x1b[91merror\x1b[0m\x1b[90m TS2345: \x1b[0mArgument of type \'{ id: number; }\' is not assignable to parameter of type \'User\'.\n  Property \'name\' is missing in type \'{ id: number; }\' but required in type \'User\'.\n\n\x1b[7m11\x1b[0m greet({ id: 2 });\n\x1b[7m  \x1b[0m \x1b[91m      ~~~~~~~~~\x1b[0m\n\n  \x1b[96msrc/app.ts\x1b[0m:\x1b[93m1\x1b[0m:\x1b[93m30\x1b[0m\n    \x1b[7m1\x1b[0m interface User { id: number; name: string; email?: string }\n    \x1b[7m \x1b[0m \x1b[96m                             ~~~~\x1b[0m\n    \'name\' is declared here.\n\n\x1b[96msrc/app.ts\x1b[0m:\x1b[93m13\x1b[0m:\x1b[93m1\x1b[0m - \x1b[91merror\x1b[0m\x1b[90m TS18047: \x1b[0m\'el\' is possibly \'null\'.\n\n\x1b[7m13\x1b[0m el.innerHTML = 3;\n\x1b[7m  \x1b[0m \x1b[91m~~\x1b[0m\n\n\x1b[96msrc/app.ts\x1b[0m:\x1b[93m13\x1b[0m:\x1b[93m1\x1b[0m - \x1b[91merror\x1b[0m\x1b[90m TS2322: \x1b[0mType \'number\' is not assignable to type \'string\'.\n\n\x1b[7m13\x1b[0m el.innerHTML = 3;\n\x1b[7m  \x1b[0m \x1b[91m~~~~~~~~~~~~\x1b[0m\n\n\nFound 8 errors in the same file, starting at: src/app.ts\x1b[90m:4\x1b[0m\n\n'),
    'diag_javac2': ('diagnostics', r"""Main.java:10: error: incompatible types: int cannot be converted to String
        xs.add(3);
               ^
Main.java:11: error: incompatible types: int cannot be converted to String
        String s = area(2, 3);
                       ^
Main.java:12: error: cannot find symbol
        undefinedMethod();
        ^
  symbol:   method undefinedMethod()
  location: class Main
Main.java:14: error: cannot find symbol
        int n = m.get("a").length();
                          ^
  symbol:   method length()
  location: class Integer
Note: Some messages have been simplified; recompile with -Xdiags:verbose to get full output
4 errors
"""),
    'diag_ruff': ('diagnostics', r"""warning: `incorrect-blank-line-before-class` (D203) and `blank-line-before-class` (D211) are incompatible. Ignoring `incorrect-blank-line-before-class`.
warning: `multi-line-summary-first-line` (D212) and `multi-line-summary-second-line` (D213) are incompatible. Ignoring `multi-line-summary-second-line`.
E401 [*] Multiple imports on one line
 --> lintme.py:1:1
  |
1 | import os, sys
  | ^^^^^^^^^^^^^^
2 | from typing import List
  |
help: Split imports
  |
  - import os, sys
1 + import os
2 + import sys
3 | from typing import List
  |

D100 Missing docstring in public module
--> lintme.py:1:1

I001 [*] Import block is un-sorted or un-formatted
 --> lintme.py:1:1
  |
1 | / import os, sys
2 | | from typing import List
  | |_______________________^
help: Organize imports
  |
  - import os, sys
1 + import os
2 + import sys
3 | from typing import List
  |

CPY001 Missing copyright notice at top of file
--> lintme.py:1:1

F401 [*] `os` imported but unused
 --> lintme.py:1:8
  |
1 | import os, sys
  |        ^^
2 | from typing import List
  |
help: Remove unused import: `os`
  |
  - import os, sys
1 + import sys
2 | from typing import List
  |

UP035 `typing.List` is deprecated, use `list` instead
 --> lintme.py:2:1
  |
1 | import os, sys
2 | from typing import List
  | ^^^^^^^^^^^^^^^^^^^^^^^

D103 Missing docstring in public function
 --> lintme.py:5:5
  |
5 | def add(a: int, b: int) -> int:
  |     ^^^
6 |     return a + b
  |

D103 Missing docstring in public function
  --> lintme.py:9:5
   |
 9 | def greet(name: str) -> str:
   |     ^^^^^
10 |     x = 1
11 |     return "hi " + name + 3
   |

F841 Local variable `x` is assigned to but never used
  --> lintme.py:10:5
   |
 9 | def greet(name: str) -> str:
10 |     x = 1
   |     ^
11 |     return "hi " + name + 3
   |
help: Remove assignment to unused variable `x`

D101 Missing docstring in public class
  --> lintme.py:14:7
   |
14 | class Box:
   |       ^^^
15 |     def __init__(self, items: List[int]):
16 |         self.items = items
   |

ANN204 Missing return type annotation for special method `__init__`
  --> lintme.py:15:9
   |
14 | class Box:
15 |     def __init__(self, items: List[int]):
"""),
    'diag_ruff_concise': ('diagnostics', r"""lintme.py:1:1: E401 [*] Multiple imports on one line
lintme.py:1:8: F401 [*] `os` imported but unused
lintme.py:2:1: UP035 `typing.List` is deprecated, use `list` instead
lintme.py:10:5: F841 Local variable `x` is assigned to but never used
lintme.py:15:31: UP006 [*] Use `list` instead of `List` for type annotation
lintme.py:24:7: F821 Undefined name `undefined_name`
lintme.py:25:1: E741 Ambiguous variable name: `l`
lintme.py:25:1: E731 Do not assign a `lambda` expression, use a `def`
lintme.py:26:17: E711 Comparison to `None` should be `cond is None`
Found 9 errors.
[*] 3 fixable with the `--fix` option (3 hidden fixes can be enabled with the `--unsafe-fixes` option).
"""),
    'diag_eslint': ('diagnostics', r"""
/home/dev/proj/ts/src/lint.js
  1:7   error    'fs' is assigned a value but never used   no-unused-vars
  2:10  error    'unused' is defined but never used        no-unused-vars
  3:3   warning  Unexpected var, use let or const instead  no-var
  3:7   error    'x' is assigned a value but never used    no-unused-vars
  4:9   warning  Expected '===' and instead saw '=='       eqeqeq
  4:26  warning  Missing semicolon                         semi
  5:3   error    Unexpected 'debugger' statement           no-debugger
  6:11  warning  Missing semicolon                         semi
  8:5   error    'y' is assigned a value but never used    no-unused-vars
  8:9   error    'undefinedVar' is not defined             no-undef

✖ 10 problems (6 errors, 4 warnings)
  0 errors and 3 warnings potentially fixable with the `--fix` option.

"""),
    'diag_go_build': ('diagnostics', r"""# example.com/demo
./main.go:5:2: "os" imported and not used
./main.go:6:2: "strings" imported and not used
./main.go:10:18: undefined: undefinedThing
./main.go:14:6: declared and not used: x
./main.go:14:14: cannot use "hello" (untyped string constant) as int value in variable declaration
./main.go:16:2: declared and not used: y
"""),
    'diag_flake8': ('diagnostics', r"""lintme.py:1:1: F401 'os' imported but unused
lintme.py:1:10: E401 multiple imports on one line
lintme.py:10:5: F841 local variable 'x' is assigned to but never used
lintme.py:24:7: F821 undefined name 'undefined_name'
lintme.py:25:1: E741 ambiguous variable name 'l'
lintme.py:25:1: E731 do not assign a lambda expression, use a def
lintme.py:25:13: E203 whitespace before ':'
lintme.py:26:3: E271 multiple spaces after keyword
lintme.py:26:14: E711 comparison to None should be 'if cond is None:'
"""),
    'log_pylogging': ('log', r"""2026-11-11 04:04:29,057 INFO     [worker.pool] processed batch 0: 443 items in 2.41s
2026-11-11 04:04:29,057 INFO     [worker.pool] processed batch 1: 70 items in 1.49s
2026-11-11 04:04:29,057 INFO     [worker.pool] processed batch 2: 343 items in 1.14s
2026-11-11 04:04:29,057 INFO     [worker.pool] processed batch 3: 259 items in 0.09s
2026-11-11 04:04:29,057 WARNING  [worker.pool] retrying job 4 (attempt 4/5): connection reset by peer
2026-11-11 04:04:29,057 DEBUG    [db.conn] query SELECT id, name FROM users WHERE org_id = 99 took 3 ms
2026-11-11 04:04:29,057 DEBUG    [db.conn] query SELECT id, name FROM users WHERE org_id = 35 took 739 ms
2026-11-11 04:04:29,057 WARNING  [worker.pool] retrying job 7 (attempt 5/5): connection reset by peer
2026-11-11 04:04:29,057 ERROR    [db.conn] deadlock detected on table orders; rolling back transaction e6c3f339
2026-11-11 04:04:29,057 INFO     [worker.pool] processed batch 9: 21 items in 0.08s
2026-11-11 04:04:29,057 DEBUG    [db.conn] query SELECT id, name FROM users WHERE org_id = 49 took 703 ms
2026-11-11 04:04:29,057 INFO     [worker.pool] processed batch 11: 226 items in 2.18s
2026-11-11 04:04:29,057 DEBUG    [db.conn] query SELECT id, name FROM users WHERE org_id = 98 took 449 ms
2026-11-11 04:04:29,057 ERROR    [db.conn] deadlock detected on table orders; rolling back transaction 8d88348a
2026-11-11 04:04:29,057 INFO     [worker.pool] processed batch 14: 128 items in 2.03s
2026-11-11 04:04:29,057 WARNING  [worker.pool] retrying job 15 (attempt 3/5): connection reset by peer
2026-11-11 04:04:29,057 ERROR    [db.conn] deadlock detected on table orders; rolling back transaction 6a8ac4ba
2026-11-11 04:04:29,057 WARNING  [worker.pool] retrying job 17 (attempt 5/5): connection reset by peer
2026-11-11 04:04:29,058 ERROR    [db.conn] deadlock detected on table orders; rolling back transaction 19999e3f
2026-11-11 04:04:29,058 INFO     [worker.pool] processed batch 19: 380 items in 2.58s
2026-11-11 04:04:29,058 INFO     [worker.pool] processed batch 20: 180 items in 2.69s
2026-11-11 04:04:29,058 ERROR    [worker.pool] job 21 failed
Traceback (most recent call last):
  File "/home/dev/proj/tb/logdemo.py", line 14, in <module>
    {}["missing_key_%d" % i]
    ~~^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'missing_key_21'
2026-11-11 04:04:29,060 DEBUG    [db.conn] query SELECT id, name FROM users WHERE org_id = 55 took 520 ms
2026-11-11 04:04:29,060 WARNING  [worker.pool] retrying job 23 (attempt 2/5): connection reset by peer
2026-11-11 04:04:29,060 INFO     [worker.pool] processed batch 24: 310 items in 2.92s
2026-11-11 04:04:29,060 INFO     [worker.pool] processed batch 25: 491 items in 1.52s
2026-11-11 04:04:29,060 DEBUG    [db.conn] query SELECT id, name FROM users WHERE org_id = 5 took 492 ms
2026-11-11 04:04:29,060 INFO     [worker.pool] processed batch 27: 418 items in 1.21s
2026-11-11 04:04:29,060 DEBUG    [db.conn] query SELECT id, name FROM users WHERE org_id = 47 took 562 ms
"""),
    'log_a1cpu': ('log', "\rLoading weights:   0%|          | 0/824 [00:00<?, ?it/s]\rLoading weights:  27%|██▋       | 222/824 [00:00<00:00, 2209.93it/s]\rLoading weights:  61%|██████▏   | 505/824 [00:00<00:00, 2565.48it/s]\rLoading weights: 100%|██████████| 824/824 [00:00<00:00, 3645.00it/s]\n[z0i] model loaded in 3s\n[z0i] rgb1s_p16: 6.8 chars/token, loss 0.152 vs blind 2.560, gap/token 2.408, per stream [2.408] (88s)\n[z0i] rgb2s_p16: 13.6 chars/token, loss 2.054 vs blind 2.810, gap/token 0.756, per stream [0.107, 1.406] (161s)\n\ninfo per vision token vs 1 stream (95% bootstrap CI):\n  p16 2 streams: R = 0.63 (0.46-0.82)\n\rLoading weights:   0%|          | 0/824 [00:00<?, ?it/s]\rLoading weights:  36%|███▌      | 296/824 [00:00<00:00, 2950.83it/s]\rLoading weights:  72%|███████▏  | 592/824 [00:00<00:00, 2876.12it/s]\rLoading weights: 100%|██████████| 824/824 [00:00<00:00, 3844.96it/s]\n[ft] rgb2s_p16 S=448 pitch=16 lines/page=28 train=1000 test=16 px2/char=57.41 chars/token=13.7 device=cpu bf16=True trainable={'trainable': 29933568, 'lora': 29933568, 'vision': 0, 'total': 3784556544} fair_hash=0e34e7db6416 pages=244s\n[ft] step 1 loss 3.6754 ema 3.6754 lr 1.60e-05 (24.91 s/step)\n[ft] step 10 loss 2.9100 ema 3.1139 lr 8.80e-05 (22.72 s/step)\n[ft] step 20 loss 2.1291 ema 2.7565 lr 1.68e-04 (23.00 s/step)\n[ft] step 30 loss 3.3466 ema 2.5564 lr 2.00e-04 (22.62 s/step)\n[ft] step 40 loss 1.8616 ema 2.3683 lr 2.00e-04 (22.90 s/step)\n[ft] step 50 loss 1.7532 ema 2.1269 lr 1.99e-04 (23.01 s/step)\n[ft] step 60 loss 0.8046 ema 1.9825 lr 1.97e-04 (23.23 s/step)\n[ft] step 70 loss 3.2060 ema 2.0568 lr 1.96e-04 (23.06 s/step)\n[ft] step 80 loss 2.3827 ema 2.0790 lr 1.93e-04 (23.05 s/step)\n[ft] step 90 loss 1.9812 ema 2.0652 lr 1.91e-04 (23.17 s/step)\n[ft] step 100 loss 3.0894 ema 2.1584 lr 1.88e-04 (23.12 s/step)\n[ft] step 110 loss 2.3047 ema 2.1796 lr 1.85e-04 (23.06 s/step)\n[ft] step 120 loss 0.6073 ema 2.0140 lr 1.81e-04 (23.02 s/step)\n[ft] step 130 loss 1.3403 ema 1.9360 lr 1.77e-04 (22.93 s/step)\n[ft] step 140 loss 1.6924 ema 1.9726 lr 1.72e-04 (22.89 s/step)\n[ft] step 150 loss 2.5374 ema 1.9280 lr 1.68e-04 (22.88 s/step)\n[ft] step 160 loss 1.8758 ema 1.8251 lr 1.63e-04 (22.92 s/step)\n[ft] step 170 loss 1.1483 ema 1.7202 lr 1.57e-04 (23.03 s/step)\n[ft] step 180 loss 1.5564 ema 1.7596 lr 1.52e-04 (23.05 s/step)\n[ft] step 190 loss 2.1819 ema 2.0721 lr 1.46e-04 (23.06 s/step)\n[ft] step 200 loss 3.1395 ema 2.3668 lr 1.40e-04 (22.94 s/step)\n[ft] step 210 loss 2.0966 ema 1.9825 lr 1.34e-04 (22.95 s/step)\n[ft] step 220 loss 0.5774 ema 1.9051 lr 1.28e-04 (22.98 s/step)\n"),
    'log_queue': ('log', r"""[queue] start job001 (ext busy=0, running=1): python3 encoder_probe/experiments/finetune_qwen.py --encoding latin_p8 --steps 1000 --run-name round3_ft_gpu3h
[queue] done rc=0: python3 encoder_probe/experiments/finetune_qwen.py --encoding latin_p8 --steps 1000 --run-name round3_ft_gpu3h
[queue] start job002 (ext busy=0, running=1): python3 encoder_probe/experiments/finetune_qwen.py --encoding rgb3_lg_pb_h5 --steps 1000 --run-name round3_ft_gpu3h
[queue] done rc=0: python3 encoder_probe/experiments/finetune_qwen.py --encoding rgb3_lg_pb_h5 --steps 1000 --run-name round3_ft_gpu3h
[queue] start job003 (ext busy=0, running=1): python3 encoder_probe/experiments/finetune_qwen.py --encoding rgb3_lg_pb_h5 --steps 1000 --run-name round3_ft_gpu3h --blind-images
[queue] done rc=0: python3 encoder_probe/experiments/finetune_qwen.py --encoding rgb3_lg_pb_h5 --steps 1000 --run-name round3_ft_gpu3h --blind-images
[queue] start job004 (ext busy=0, running=1): python3 encoder_probe/experiments/finetune_qwen.py --encoding latin_p8 --steps 1000 --run-name round3_ft_gpu3h --blind-images
[queue] done rc=0: python3 encoder_probe/experiments/finetune_qwen.py --encoding latin_p8 --steps 1000 --run-name round3_ft_gpu3h --blind-images
[queue] start job005 (ext busy=0, running=1): python3 encoder_probe/experiments/finetune_qwen.py --encoding latin_p5 --steps 1000 --run-name round3_ft_gpu3h
[queue] done rc=0: python3 encoder_probe/experiments/finetune_qwen.py --encoding latin_p5 --steps 1000 --run-name round3_ft_gpu3h
[queue] start job006 (ext busy=0, running=1): python3 encoder_probe/experiments/finetune_qwen.py --encoding lg_pb_h5 --steps 1000 --run-name round3_ft_gpu3h
[queue] done rc=0: python3 encoder_probe/experiments/finetune_qwen.py --encoding lg_pb_h5 --steps 1000 --run-name round3_ft_gpu3h
[queue] empty, exiting
"""),
    'test_cargo': ('log', r"""   Compiling rst v0.1.0 (/home/dev/proj/rst)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 2.94s
     Running unittests src/lib.rs (target/debug/deps/rst-d47e89702ea842f8)

running 4 tests
test tests::adds ... ok
test tests::adds_wrong ... FAILED
test tests::divides_by_zero ... FAILED
test tests::strings ... FAILED

failures:

---- tests::adds_wrong stdout ----

thread 'tests::adds_wrong' (3065) panicked at src/lib.rs:10:23:
assertion `left == right` failed: math is broken for 2
  left: 4
 right: 5
stack backtrace:
   0: __rustc::rust_begin_unwind
             at /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/std/src/panicking.rs:689:5
   1: core::panicking::panic_fmt
             at /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/core/src/panicking.rs:80:14
   2: core::panicking::assert_failed_inner
             at /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/core/src/panicking.rs:434:23
   3: core::panicking::assert_failed::<i32, i32>
             at /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/core/src/panicking.rs:394:5
   4: rst::tests::adds_wrong
             at ./src/lib.rs:10:23
   5: rst::tests::adds_wrong::{closure#0}
             at ./src/lib.rs:10:20
   6: <rst::tests::adds_wrong::{closure#0} as core::ops::function::FnOnce<()>>::call_once
             at /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/core/src/ops/function.rs:250:5
   7: <fn() -> core::result::Result<(), alloc::string::String> as core::ops::function::FnOnce<()>>::call_once
             at /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/core/src/ops/function.rs:250:5
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.

---- tests::divides_by_zero stdout ----

thread 'tests::divides_by_zero' (3066) panicked at src/lib.rs:2:37:
attempt to divide by zero
stack backtrace:
   0: __rustc::rust_begin_unwind
             at /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/std/src/panicking.rs:689:5
   1: core::panicking::panic_fmt
             at /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/core/src/panicking.rs:80:14
   2: core::panicking::panic_const::panic_const_div_by_zero
             at /rustc/2d8144b7880597b6e6d3dfd63a9a9efae3f533d3/library/core/src/panicking.rs:175:17
   3: rst::div
             at ./src/lib.rs:2:37
   4: rst::tests::divides_by_zero
             at ./src/lib.rs:12:28
   5: rst::tests::divides_by_zero::{closure#0}
             at ./src/lib.rs:12:25
"""),
    'test_go_v': ('log', r"""=== RUN   TestAdd
    calc_test.go:10: Add(2, 2) = 4; want 5
--- FAIL: TestAdd (0.00s)
=== RUN   TestTable
=== RUN   TestTable/1+1
=== RUN   TestTable/2+3
    calc_test.go:19: got 5, want 6
=== RUN   TestTable/0+0
    calc_test.go:19: got 0, want 1
--- FAIL: TestTable (0.00s)
    --- PASS: TestTable/1+1 (0.00s)
    --- FAIL: TestTable/2+3 (0.00s)
    --- FAIL: TestTable/0+0 (0.00s)
=== RUN   TestDiv
--- FAIL: TestDiv (0.00s)
panic: runtime error: integer divide by zero [recovered]
	panic: runtime error: integer divide by zero

goroutine 11 [running]:
testing.tRunner.func1.2({0x54ff00, 0x6a1870})
	/usr/local/go1.24.7/src/testing/testing.go:1734 +0x21c
testing.tRunner.func1()
	/usr/local/go1.24.7/src/testing/testing.go:1737 +0x35e
panic({0x54ff00?, 0x6a1870?})
	/usr/local/go1.24.7/src/runtime/panic.go:792 +0x132
example.com/demo.Div(...)
	/home/dev/proj/got/calc.go:4
example.com/demo.TestDiv(0xc00013a540?)
	/home/dev/proj/got/calc_test.go:26 +0xa
testing.tRunner(0xc00013a540, 0x5845a8)
	/usr/local/go1.24.7/src/testing/testing.go:1792 +0xf4
created by testing.(*T).Run in goroutine 1
	/usr/local/go1.24.7/src/testing/testing.go:1851 +0x413
FAIL	example.com/demo	0.004s
FAIL
"""),
    'test_npm': ('log', r"""
> nodet@1.0.0 test
> node --test

TAP version 13
# Subtest: adds numbers
ok 1 - adds numbers
  ---
  duration_ms: 0.606387
  type: 'test'
  ...
# Subtest: adds wrong
not ok 2 - adds wrong
  ---
  duration_ms: 1.163539
  type: 'test'
  location: '/home/dev/proj/nodet/math.test.js:5:1'
  failureType: 'testCodeFailure'
  error: |-
    Expected values to be strictly equal:
    
    3 !== 4
    
  code: 'ERR_ASSERTION'
  name: 'AssertionError'
  expected: 4
  actual: 3
  operator: 'strictEqual'
  stack: |-
    TestContext.<anonymous> (/home/dev/proj/nodet/math.test.js:5:35)
    Test.runInAsyncScope (node:async_hooks:214:14)
    Test.run (node:internal/test_runner/test:1047:25)
    Test.processPendingSubtests (node:internal/test_runner/test:744:18)
    Test.postRun (node:internal/test_runner/test:1173:19)
    Test.run (node:internal/test_runner/test:1101:12)
    async startSubtestAfterBootstrap (node:internal/test_runner/harness:296:3)
  ...
# Subtest: deep equal
not ok 3 - deep equal
  ---
  duration_ms: 1.089668
  type: 'test'
  location: '/home/dev/proj/nodet/math.test.js:6:1'
  failureType: 'testCodeFailure'
  error: |-
    Expected values to be strictly deep-equal:
    + actual - expected
    
      {
        a: 1,
        b: [
          1,
    +     2
    -     3
        ]
      }
    
  code: 'ERR_ASSERTION'
  name: 'AssertionError'
  expected:
    a: 1
    b:
      0: 1
      1: 3
  actual:
    a: 1
    b:
      0: 1
      1: 2
  operator: 'deepStrictEqual'
  stack: |-
    TestContext.<anonymous> (/home/dev/proj/nodet/math.test.js:6:35)
    Test.runInAsyncScope (node:async_hooks:214:14)
    Test.run (node:internal/test_runner/test:1047:25)
    Test.processPendingSubtests (node:internal/test_runner/test:744:18)
    Test.postRun (node:internal/test_runner/test:1173:19)
    Test.run (node:internal/test_runner/test:1101:12)
    async Test.processPendingSubtests (node:internal/test_runner/test:744:7)
  ...
# Subtest: throws
not ok 4 - throws
  ---
  duration_ms: 0.161295
  type: 'test'
  location: '/home/dev/proj/nodet/math.test.js:7:1'
  failureType: 'testCodeFailure'
  error: "Expected property name or '}' in JSON at position 1 (line 1 column 2)"
"""),
    'gitdiff_head': ('gitdiff', 'diff --git a/blob.bin b/blob.bin\nnew file mode 100644\nindex 0000000..f541187\nBinary files /dev/null and b/blob.bin differ\ndiff --git a/src/app.py b/src/app.py\nindex 237918a..ae3422c 100644\n--- a/src/app.py\n+++ b/src/app.py\n@@ -5,6 +5,7 @@ class Config:\n     def __init__(self, path):\n         self.path = path\n         self.data = None\n+        self.cache = {}\n \n     def load(self):\n         with open(self.path) as f:\n@@ -12,17 +13,22 @@ class Config:\n         return self\n \n     def get(self, key):\n-        return self.data["settings"][key]\n+        if key in self.cache:\n+            return self.cache[key]\n+        value = self.data["settings"][key]\n+        self.cache[key] = value\n+        return value\n \n \n def process(items, scale):\n-    total = 0\n+    total  = 0.0\n     for i, item in enumerate(items):\n         total += item["value"] * scale / item["weight"]\n     return total\n \n \n-def main():\n+def main(argv=None):\n+    """Entry point."""\n     cfg = Config("missing.json")\n     try:\n         cfg.load()\n@@ -53,4 +59,4 @@ if __name__ == "__main__":\n         raise ExceptionGroup("many errors", [ValueError("bad value"), TypeError("bad type")])\n     elif which == "attr":\n         import os\n-        os.path.joinn("a", "b")\n+        os.path.joinn("a", "b")\n\\ No newline at end of file\ndiff --git a/tabs.txt b/tabs.txt\nnew file mode 100644\nindex 0000000..94d410c\n--- /dev/null\n+++ b/tabs.txt\n@@ -0,0 +1 @@\n+\tindented with tab\n'),
    'gitdiff_cached': ('gitdiff', r"""diff --git a/blob.bin b/blob.bin
new file mode 100644
index 0000000..f541187
Binary files /dev/null and b/blob.bin differ
diff --git a/tabs.txt b/tabs.txt
new file mode 100644
index 0000000..94d410c
--- /dev/null
+++ b/tabs.txt
@@ -0,0 +1 @@
+	indented with tab
"""),
    'diffu_app': ('gitdiff', '--- /home/dev/proj/tb/app.py\t2026-11-11 04:00:45.885617895 +0000\n+++ src/app.py\t2026-11-11 04:04:43.454352331 +0000\n@@ -5,6 +5,7 @@\n     def __init__(self, path):\n         self.path = path\n         self.data = None\n+        self.cache = {}\n \n     def load(self):\n         with open(self.path) as f:\n@@ -12,17 +13,22 @@\n         return self\n \n     def get(self, key):\n-        return self.data["settings"][key]\n+        if key in self.cache:\n+            return self.cache[key]\n+        value = self.data["settings"][key]\n+        self.cache[key] = value\n+        return value\n \n \n def process(items, scale):\n-    total = 0\n+    total  = 0.0\n     for i, item in enumerate(items):\n         total += item["value"] * scale / item["weight"]\n     return total\n \n \n-def main():\n+def main(argv=None):\n+    """Entry point."""\n     cfg = Config("missing.json")\n     try:\n         cfg.load()\n@@ -53,4 +59,4 @@\n         raise ExceptionGroup("many errors", [ValueError("bad value"), TypeError("bad type")])\n     elif which == "attr":\n         import os\n-        os.path.joinn("a", "b")\n+        os.path.joinn("a", "b")\n\\ No newline at end of file\n'),
    'gitshow_s': ('gitdiff', 'commit ddbb4f3de680c07dcc4bdc929d9068368f7d058a\nAuthor: Dev <dev@example.com>\nDate:   Wed Nov 11 04:18:06 2026 +0000\n\n    Config cache; float totals; main takes argv\n    \n    Config.get now memoises settings lookups so repeated calls do not walk the\n    nested dict again. process() accumulates in float.\n    \n    Refs: #42\n\ndiff --git a/blob.bin b/blob.bin\nnew file mode 100644\nindex 0000000..f541187\nBinary files /dev/null and b/blob.bin differ\ndiff --git a/src/app.py b/src/app.py\nindex 237918a..ae3422c 100644\n--- a/src/app.py\n+++ b/src/app.py\n@@ -5,6 +5,7 @@ class Config:\n     def __init__(self, path):\n         self.path = path\n         self.data = None\n+        self.cache = {}\n \n     def load(self):\n         with open(self.path) as f:\n@@ -12,17 +13,22 @@ class Config:\n         return self\n \n     def get(self, key):\n-        return self.data["settings"][key]\n+        if key in self.cache:\n+            return self.cache[key]\n+        value = self.data["settings"][key]\n+        self.cache[key] = value\n+        return value\n \n \n def process(items, scale):\n-    total = 0\n+    total  = 0.0\n     for i, item in enumerate(items):\n         total += item["value"] * scale / item["weight"]\n     return total\n \n \n-def main():\n+def main(argv=None):\n+    """Entry point."""\n     cfg = Config("missing.json")\n     try:\n         cfg.load()\n@@ -53,4 +59,4 @@ if __name__ == "__main__":\n         raise ExceptionGroup("many errors", [ValueError("bad value"), TypeError("bad type")])\n     elif which == "attr":\n         import os\n-        os.path.joinn("a", "b")\n+        os.path.joinn("a", "b")\n\\ No newline at end of file\ndiff --git a/tabs.txt b/tabs.txt\nnew file mode 100644\nindex 0000000..94d410c\n--- /dev/null\n+++ b/tabs.txt\n@@ -0,0 +1 @@\n+\tindented with tab\n'),
    'gitlogp_s': ('gitdiff', r"""commit 09c509013513d5eb27c8a164903262b86a791c91
Author: Ana <ana@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    notes: entry 3

diff --git a/notes.txt b/notes.txt
index 1bf1d7b..038b43a 100644
--- a/notes.txt
+++ b/notes.txt
@@ -1,3 +1,4 @@
 n
 v1
 v2
+v3

commit 77f9dc3d1ed4110ae672e7bb5a277a37368301d0
Author: Ana <ana@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    notes: entry 2

diff --git a/notes.txt b/notes.txt
index 7a6d66a..1bf1d7b 100644
--- a/notes.txt
+++ b/notes.txt
@@ -1,2 +1,3 @@
 n
 v1
+v2
"""),
    'gitdiff_s_stat': ('gitdiff', r""" blob.bin       | Bin 0 -> 300 bytes
 docs/readme.md |   1 +
 notes.txt      |   3 +++
 src/app.py     |  14 ++++++++++----
 src/helper.py  |   2 ++
 tabs.txt       |   1 +
 6 files changed, 17 insertions(+), 4 deletions(-)
"""),
    'gitlog_s_default': ('gitlog', r"""commit 09c509013513d5eb27c8a164903262b86a791c91
Author: Ana <ana@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    notes: entry 3

commit 77f9dc3d1ed4110ae672e7bb5a277a37368301d0
Author: Ana <ana@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    notes: entry 2

commit c4488228ad6da1b7d1d83e7b76bef01918b2f745
Author: Ana <ana@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    notes: entry 1

commit 34197e7e96757c6e0e63a34db94dac4f38a4825b
Merge: 6983822 fe5abe5
Author: Dev <dev@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    Merge branch 'feature'

commit 69838228cbe2ba38a8735ca1071f90f9c59e75e9
Author: Ana <ana@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    docs: heading

commit fe5abe5801a824e83325cc6f5cdb78b696926599
Author: Dev <dev@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    Add helper module

commit ddbb4f3de680c07dcc4bdc929d9068368f7d058a
Author: Dev <dev@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    Config cache; float totals; main takes argv
    
    Config.get now memoises settings lookups so repeated calls do not walk the
    nested dict again. process() accumulates in float.
    
    Refs: #42

commit e0988cf092dd1813b1106efc304303b97130bf27
Author: Test <t@example.com>
Date:   Wed Nov 11 04:04:43 2026 +0000

    app

commit 8b5dae954779266d58081810aeea02750e800029
Author: Test <t@example.com>
Date:   Wed Nov 11 04:04:43 2026 +0000

    second

commit cfa2352d1952d3db1a1b89cad2f0a6a8595e3bcf
Author: Test <t@example.com>
Date:   Wed Nov 11 04:04:35 2026 +0000

    init
"""),
    'gitlog_s_oneline': ('gitlog', r"""09c5090 notes: entry 3
77f9dc3 notes: entry 2
c448822 notes: entry 1
34197e7 Merge branch 'feature'
6983822 docs: heading
fe5abe5 Add helper module
ddbb4f3 Config cache; float totals; main takes argv
e0988cf app
8b5dae9 second
cfa2352 init
"""),
    'gitlog_s_stat': ('gitlog', r"""commit 09c509013513d5eb27c8a164903262b86a791c91
Author: Ana <ana@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    notes: entry 3

 notes.txt | 1 +
 1 file changed, 1 insertion(+)

commit 77f9dc3d1ed4110ae672e7bb5a277a37368301d0
Author: Ana <ana@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    notes: entry 2

 notes.txt | 1 +
 1 file changed, 1 insertion(+)

commit c4488228ad6da1b7d1d83e7b76bef01918b2f745
Author: Ana <ana@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    notes: entry 1

 notes.txt | 1 +
 1 file changed, 1 insertion(+)

commit 34197e7e96757c6e0e63a34db94dac4f38a4825b
Merge: 6983822 fe5abe5
Author: Dev <dev@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    Merge branch 'feature'
"""),
    'gitlog_s_graph': ('gitlog', "* 09c5090 notes: entry 3\n* 77f9dc3 notes: entry 2\n* c448822 notes: entry 1\n*   34197e7 Merge branch 'feature'\n|\\  \n| * fe5abe5 Add helper module\n* | 6983822 docs: heading\n|/  \n* ddbb4f3 Config cache; float totals; main takes argv\n* e0988cf app\n* 8b5dae9 second\n* cfa2352 init\n"),
    'gitshow_s_merge': ('gitlog', r"""commit 34197e7e96757c6e0e63a34db94dac4f38a4825b
Merge: 6983822 fe5abe5
Author: Dev <dev@example.com>
Date:   Wed Nov 11 04:18:06 2026 +0000

    Merge branch 'feature'

 src/helper.py | 2 ++
 1 file changed, 2 insertions(+)
"""),
    'gitstatus_long': ('gitstatus', r"""On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
	renamed:    old.txt -> config.cfg
	modified:   docs/readme.md
	deleted:    setup.cfg
	modified:   src/pkg/c.py

Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   src/a.py
	deleted:    src/b.py
	modified:   src/pkg/c.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	build/
	notes.txt
	src/new_module.py

"""),
    'gitstatus_short': ('gitstatus', r"""R  old.txt -> config.cfg
M  docs/readme.md
D  setup.cfg
 M src/a.py
 D src/b.py
MM src/pkg/c.py
?? build/
?? notes.txt
?? src/new_module.py
"""),
    'gitstatus_sb': ('gitstatus', r"""## main
R  old.txt -> config.cfg
M  docs/readme.md
D  setup.cfg
 M src/a.py
 D src/b.py
MM src/pkg/c.py
?? build/
?? notes.txt
?? src/new_module.py
"""),
    'grep_rn_render': ('grep', r"""./encoder_probe/ep/pages.py:84:    def render(self, text: str, pos: int) -> tuple[Page, int]:  # pragma: no cover - interface
./encoder_probe/ep/pages.py:103:    def render(self, text: str, pos: int) -> tuple[Page, int]:
./encoder_probe/ep/pages.py:139:    def render(self, text: str, pos: int) -> tuple[Page, int]:
./encoder_probe/ep/rgb_overlay.py:101:    def render(self, text: str, pos: int) -> tuple[Page, int]:
./encoder_probe/ep/rgb_overlay.py:156:    def render(self, text: str, pos: int) -> tuple[Page, int]:
./encoder_probe/ep/rgb_overlay.py:243:    def render(self, text: str, pos: int) -> tuple[Page, int]:
./encoder_probe/ep/rgb_overlay.py:322:    def render(self, text: str, pos: int) -> tuple[Page, int]:
./encoder_probe/ep/geom.py:64:    def render(self, text: str, pos: int) -> tuple[Page, int]:
./encoder_probe/ep/geom.py:90:    def render(self, text: str, pos: int) -> tuple[Page, int]:
./encoder_probe/ep/geom.py:128:    def render(self, text: str, pos: int) -> tuple[Page, int]:
./encoder_probe/ep/geom.py:153:    def render(self, text: str, pos: int) -> tuple[Page, int]:
./encoder_probe/ep/geom.py:209:    def render(self, text: str, pos: int) -> tuple[Page, int]:
./encoder_probe/experiments/longctx_bench.py:74:def render_document(enc, doc: str) -> list[np.ndarray]:
./learned_glyphs/lg/glyphs.py:71:def render_font_cells(symbols: Sequence[str], font_path, height_px: int, ss: int = 4,
./learned_glyphs/lg/train.py:555:def render(ink, ids_batch, widths_ss_t, compose, channel, cfg, generator, train: bool,
./experiments/claude_qa_render.py:299:def render_pages(doc: str, budget: float, page_px: int = 0) -> tuple[list, list[dict]]:
./experiments/readhook/vtchook/render.py:57:def render(flow: str, budget: float = DEFAULT_BUDGET) -> list[Page]:
./experiments/readhook/render_code.py:6:def render(lines, first_no, out, W=1932, H=1876, em=13.8, col_chars=100, header=""):
./vtc/layout.py:188:def render_page(units: list[Unit], W: int, H: int, p: LayoutParams | None = None,
./vtc/layout.py:208:def render_line(units: list[Unit], em: float, p: LayoutParams | None = None,
./vtc/crnn/data.py:181:def render_sample(scheme, line: str, em, rp: RenderParams, rng: np.random.Generator,
./vtc/crnn/train.py:54:    def render_params(self) -> D.RenderParams:
"""),
    'grep_ctx_scheme': ('grep', 'vtc/schemes/substitution.py-123-\nvtc/schemes/substitution.py:124:class SubstitutionScheme(Scheme):\nvtc/schemes/substitution.py-125-    def __init__(self, n_tokens: int = 300, dictionary=None, legend: bool = False,\nvtc/schemes/substitution.py-126-                 leftover_font: str | Path = config.CONDENSED_FONT,\nvtc/schemes/substitution.py-127-                 cjk_font: str | Path = config.CJK_FONT, cjk_scale: float = 1.0,\n--\nvtc/schemes/base.py-17-\nvtc/schemes/base.py:18:class Scheme(ABC):\nvtc/schemes/base.py-19-    name: str = "base"\nvtc/schemes/base.py-20-\nvtc/schemes/base.py-21-    @abstractmethod\n--\nvtc/schemes/latin.py-10-\nvtc/schemes/latin.py:11:class LatinScheme(Scheme):\nvtc/schemes/latin.py-12-    def __init__(self, font_path: str | Path = config.LATIN_FONT, name: str = "latin"):\nvtc/schemes/latin.py-13-        self.font_path = str(font_path)\nvtc/schemes/latin.py-14-        self.name = name\n--\nvtc/schemes/quadrant.py-190-\nvtc/schemes/quadrant.py:191:class QuadrantScheme(Scheme):\nvtc/schemes/quadrant.py-192-    """Hangul-style packing of k = 2, 3 or 4 characters per cell.\nvtc/schemes/quadrant.py-193-\nvtc/schemes/quadrant.py-194-    ``cell_width_em=None`` (default) uses *proportional* cells: each slot column is as wide as the\n--\nvtc/crnn/stubs.py-40-\nvtc/crnn/stubs.py:41:class StubCJK(Scheme):\nvtc/crnn/stubs.py-42-    def __init__(self):\nvtc/crnn/stubs.py-43-        self.name = "stub_cjk"\nvtc/crnn/stubs.py-44-        self.pairs = {a + b: chr(0x4E00 + i) for i, (a, b) in\n--\nvtc/crnn/stubs.py-63-\nvtc/crnn/stubs.py:64:class StubQuad(Scheme):\nvtc/crnn/stubs.py-65-    def __init__(self):\nvtc/crnn/stubs.py-66-        self.name = "stub_quad"\nvtc/crnn/stubs.py-67-\n'),
    'rg_C2_fitem': ('grep', 'vtc/layout.py-148-\nvtc/layout.py-149-\nvtc/layout.py:150:def fit_em(units: list[Unit], W: int, H: int, p: LayoutParams | None = None,\nvtc/layout.py-151-           lo: float = 1.0, hi: float | None = None, iters: int = 22) -> float:\nvtc/layout.py-152-    """Largest em (to ~0.01 px) at which ``units`` fit in W x H. Returns 0.0 if even ``lo`` fails."""\n'),
    'grep_n_single': ('grep', r"""30:def get_font(path: str | Path, size: float) -> ImageFont.FreeTypeFont:
35:def glyph_advance(path: str, size: float, text: str) -> float:
54:    def advance(self, em: float) -> float:  # pragma: no cover - interface
57:    def draw(self, d: ImageDraw.ImageDraw, x: float, baseline: float, em: float) -> None:  # pragma: no cover
66:    def __init__(self, char: str, font_path: str | Path, source: str | None = None,
74:    def advance(self, em: float) -> float:
77:    def draw(self, d: ImageDraw.ImageDraw, x: float, baseline: float, em: float) -> None:
88:    def pitch(self, em: float) -> float:
91:    def first_baseline(self, em: float) -> float:
95:def _words(units: list[Unit]) -> list[list[Unit]]:
112:def wrap(units: list[Unit], em: float, max_width: float) -> list[list[Unit]] | None:
141:def fits(units: list[Unit], em: float, W: int, H: int, p: LayoutParams) -> list[list[Unit]] | None:
150:def fit_em(units: list[Unit], W: int, H: int, p: LayoutParams | None = None,
168:def _draw_lines(lines: list[list[Unit]], em: float, size: tuple[int, int], p: LayoutParams,
188:def render_page(units: list[Unit], W: int, H: int, p: LayoutParams | None = None,
208:def render_line(units: list[Unit], em: float, p: LayoutParams | None = None,
"""),
    'rg_column': ('grep', r"""vtc/dictionary.py:159:15:        rec = json.loads(Path(path).read_text())
vtc/crnn/data.py:265:16:        meta = json.loads((d / "meta.json").read_text())
vtc/analysis.py:70:33:                    rows.append(json.loads(line))
vtc/corpus.py:263:17:        return [json.loads(line) for line in f if line.strip()]
vtc/claude_eval.py:167:18:            ev = json.loads(line)
vtc/claude_eval.py:250:19:            rec = json.loads(path.read_text())
tests/test_crnn.py:162:13:    rows = [json.loads(x) for x in open(tmp_path / "runs.jsonl")]
tests/test_crnn.py:224:12:    pts = [json.loads(x) for x in open(tmp_path / "curves" / "latin_mixed5-8.jsonl")]
tests/test_analysis.py:179:10:    js = json.loads((tmp_path / "summary.json").read_text())
tests/test_analysis.py:208:13:    rows = [json.loads(x) for x in (run / "runs.jsonl").read_text().splitlines()]
tests/test_claude_eval.py:83:11:    msg = json.loads(call["stdin"])
tests/test_claude_eval.py:208:18:        prompt = json.loads(stdin)["message"]["content"][-1]["text"]
scripts/merge_phase2.py:17:13:        r = json.loads(line)
experiments/readhook/smoke/run_smoke.py:52:11:    ev = [json.loads(l) for l in p.stdout.splitlines() if l.strip().startswith("{")]
experiments/readhook/smoke/run_smoke.py:72:13:    hook = [json.loads(l) for l in (base / "hook.jsonl").read_text().splitlines()] if (base / "hook.jsonl").exists() else []
scripts/check_claude_vision.py:193:13:        d = json.loads(p.read_text())
scripts/check_claude_vision.py:215:17:            d = json.loads(p.read_text())
scripts/make_readme_figures.py:526:21:            out += [json.loads(line) for line in f if line.strip()]
scripts/make_claude_qa_figures.py:78:16:    return {m: json.loads(v["summary"].read_text()) for m, v in MODELS.items()}
scripts/make_claude_qa_figures.py:339:17:            r = json.loads(line)
scripts/make_claude_qa_figures.py:552:17:        rows = {json.loads(x)["arm"]: json.loads(x) for x in f.read_text().splitlines() if x.strip()}
"""),
    'rg_nonum_init': ('grep', r"""vtc/dictionary.py:    def __init__(self, tokens: list[str], font_path: str | Path = config.CONDENSED_FONT,
vtc/dictionary.py:    def __init__(self, text: str, font_path: str, symbol_em: float, log=print):
vtc/crnn/stubs.py:    def __init__(self, chars: str, font_path: str):
vtc/crnn/stubs.py:    def __init__(self):
vtc/crnn/stubs.py:    def __init__(self):
vtc/crnn/data.py:    def __init__(self, phase: float, em: float):
vtc/crnn/data.py:    def __init__(self, images: np.ndarray, offsets: np.ndarray, labels: list[tuple[str, ...]],
vtc/crnn/data.py:    def __init__(self, symbols: Sequence[str]):
vtc/crnn/model.py:    def __init__(self, cap: int):
vtc/crnn/model.py:    def __init__(self, n_classes: int, col_stride: int = 2, expand: int = 1, channels=(16, 64, 96),
vtc/analysis.py:    def __init__(self, rows: list[dict], scheme: str, units: list | None = None):
vtc/layout.py:    def __init__(self, char: str, font_path: str | Path, source: str | None = None,
vtc/schemes/quadrant.py:    def __init__(self, chunk: str, scheme: "QuadrantScheme"):
vtc/schemes/quadrant.py:    def __init__(self, char: str, font_path: str, width_em: float):
vtc/schemes/quadrant.py:    def __init__(self, k: int = 3, font_path: str | Path = DEFAULT_FONT,
vtc/schemes/latin.py:    def __init__(self, font_path: str | Path = config.LATIN_FONT, name: str = "latin"):
vtc/schemes/substitution.py:    def __init__(self, char: str, font_path: str | Path, source: str, scale: float = 1.0,
"""),
    'rg_C1_subprocess': ('grep', 'vtc/claude_eval.py-6-  * no tools, no MCP servers, no slash commands/skills, no settings files, no session saved;\nvtc/claude_eval.py:7:  * the subprocess runs in an empty temp dir (no CLAUDE.md / project context);\nvtc/claude_eval.py-8-  * the environment is a small whitelist, so variables of a *parent* Claude Code session\n--\nvtc/claude_eval.py-29-import re\nvtc/claude_eval.py:30:import subprocess\nvtc/claude_eval.py-31-import sys\n--\nvtc/claude_eval.py-146-        try:\nvtc/claude_eval.py:147:            p = subprocess.run(cmd, input=stdin, capture_output=True, env=env, cwd=cwd, timeout=timeout)\nvtc/claude_eval.py:148:        except subprocess.TimeoutExpired as e:\nvtc/claude_eval.py-149-            out = (e.stdout or b"").decode("utf-8", "replace")\n--\nscripts/job_queue.py-14-import os\nscripts/job_queue.py:15:import subprocess\nscripts/job_queue.py-16-import sys\n--\nscripts/job_queue.py-42-    claimed = 0\nscripts/job_queue.py:43:    running: dict[int, tuple[subprocess.Popen, str]] = {}\nscripts/job_queue.py-44-    idle_rounds = 0\n--\nscripts/job_queue.py-72-                log.flush()\nscripts/job_queue.py:73:                proc = subprocess.Popen(cmd, shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,\nscripts/job_queue.py-74-                                        env={**os.environ, "OMP_NUM_THREADS": "1"})\n--\ntests/test_claude_eval.py:1:"""Unit tests for vtc.claude_eval. The CLI subprocess is monkeypatched: no real Claude calls."""\ntests/test_claude_eval.py-2-from __future__ import annotations\n--\nexperiments/readhook/smoke/run_smoke.py-9-import shutil\nexperiments/readhook/smoke/run_smoke.py:10:import subprocess\nexperiments/readhook/smoke/run_smoke.py-11-import sys\n--\nexperiments/readhook/smoke/run_smoke.py-50-           "--strict-mcp-config", "--no-session-persistence", "--max-turns", "40"]\nexperiments/readhook/smoke/run_smoke.py:51:    p = subprocess.run(cmd, capture_output=True, text=True, env=CE.build_env(), cwd=work, timeout=3600)\nexperiments/readhook/smoke/run_smoke.py-52-    ev = [json.loads(l) for l in p.stdout.splitlines() if l.strip().startswith("{")]\n--\nexperiments/readhook/smoke/run_smoke.py-72-    hook = [json.loads(l) for l in (base / "hook.jsonl").read_text().splitlines()] if (base / "hook.jsonl").exists() else []\nexperiments/readhook/smoke/run_smoke.py:73:    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=work,\n'),
    'find_py': ('find', r"""vtc/config.py
vtc/__init__.py
vtc/metrics.py
vtc/readers/__init__.py
vtc/readers/rapidocr_reader.py
vtc/claude_eval.py
vtc/schemes/substitution.py
vtc/schemes/base.py
vtc/schemes/__init__.py
vtc/schemes/latin.py
vtc/schemes/quadrant.py
vtc/layout.py
vtc/corpus.py
vtc/analysis.py
vtc/crnn/model.py
vtc/crnn/__init__.py
vtc/crnn/data.py
vtc/crnn/stubs.py
vtc/crnn/train.py
vtc/dictionary.py
encoder_probe/ep/g2lite.py
encoder_probe/ep/pages.py
encoder_probe/ep/token_cache.py
encoder_probe/ep/qwen_ft.py
encoder_probe/ep/__init__.py
encoder_probe/ep/deepencoder.py
encoder_probe/ep/r3_tasks.py
encoder_probe/ep/vlm_encoders.py
encoder_probe/ep/encoders.py
encoder_probe/ep/qwen_split.py
encoder_probe/ep/flock.py
encoder_probe/ep/hf_partial.py
encoder_probe/ep/rgb_overlay.py
encoder_probe/ep/probe_grid.py
encoder_probe/ep/probe.py
encoder_probe/ep/geom.py
encoder_probe/__init__.py
encoder_probe/tests/test_text_compress.py
encoder_probe/tests/test_rgb_overlay.py
encoder_probe/tests/test_g2lite.py
encoder_probe/tests/test_r3_0b.py
encoder_probe/tests/test_capacity_curve.py
encoder_probe/tests/test_rgb3_lg.py
encoder_probe/tests/test_analyse_r3.py
encoder_probe/tests/test_round3.py
encoder_probe/tests/test_round2.py
encoder_probe/tests/test_probe.py
encoder_probe/tests/test_colour.py
encoder_probe/tests/test_channel_capacity.py
encoder_probe/experiments/analyse_g2lite.py
encoder_probe/experiments/capacity_curve.py
encoder_probe/experiments/run_probe.py
encoder_probe/experiments/run12_driver.py
encoder_probe/experiments/analyse.py
encoder_probe/experiments/small_vlm.py
encoder_probe/experiments/text_compress.py
encoder_probe/experiments/channel_capacity.py
encoder_probe/experiments/zeroshot_colour.py
encoder_probe/experiments/make_figures.py
encoder_probe/experiments/quick_linear_readout.py
encoder_probe/experiments/run_g2lite.py
encoder_probe/experiments/longctx_bench.py
encoder_probe/experiments/analyse_run12.py
encoder_probe/experiments/g2lite_decode.py
encoder_probe/experiments/analyse_r3.py
encoder_probe/experiments/zeroshot_info.py
encoder_probe/experiments/analyse_colour.py
encoder_probe/experiments/build_tokens2.py
encoder_probe/experiments/run_probe2.py
encoder_probe/experiments/analyse2.py
encoder_probe/experiments/analyse_stage0.py
encoder_probe/experiments/finetune_qwen.py
encoder_probe/results/run0b_cpu_p6/decompose_dxdy.py
"""),
    'find_exp': ('find', r"""experiments
experiments/phase1b_ocr.py
experiments/phase2_driver.py
experiments/phase1_sweep.py
experiments/claude_qa_research.py
experiments/__init__.py
experiments/claude_qa_render.py
experiments/phase2_main.sh
experiments/analyse_claude_qa.py
experiments/phase2_train.py
experiments/readhook
experiments/readhook/list.txt
experiments/readhook/vtchook
experiments/readhook/vtchook/minify
experiments/readhook/vtchook/common.py
experiments/readhook/vtchook/read_hook.py
experiments/readhook/vtchook/render.py
experiments/readhook/vtchook/edit_hook.py
experiments/readhook/vtchook/__init__.py
experiments/readhook/vtchook/SPEC.md
experiments/readhook/vtchook/install.py
experiments/readhook/vtchook/tests
experiments/readhook/vtchook/policy.py
experiments/readhook/vtchook/bash_hook.py
experiments/readhook/bash_output_hook.py
experiments/readhook/README.md
experiments/readhook/javamin.py
experiments/readhook/render_code.py
experiments/readhook/smoke
experiments/readhook/smoke/img_read_hook.py
experiments/readhook/smoke/tasks
experiments/readhook/smoke/img_edit_hook.py
experiments/readhook/smoke/run_smoke.py
experiments/readhook/smoke/bash_guard_hook.py
experiments/readhook/smoke/log_read_hook.py
"""),
    'rg_l': ('find', r"""vtc/analysis.py
encoder_probe/experiments/finetune_qwen.py
encoder_probe/experiments/analyse_stage0.py
encoder_probe/experiments/analyse2.py
encoder_probe/experiments/run_probe2.py
encoder_probe/experiments/build_tokens2.py
encoder_probe/experiments/analyse_colour.py
encoder_probe/experiments/zeroshot_info.py
encoder_probe/experiments/analyse_r3.py
encoder_probe/experiments/g2lite_decode.py
encoder_probe/experiments/analyse_run12.py
encoder_probe/experiments/longctx_bench.py
encoder_probe/experiments/run_g2lite.py
encoder_probe/experiments/quick_linear_readout.py
encoder_probe/experiments/zeroshot_colour.py
encoder_probe/experiments/channel_capacity.py
encoder_probe/experiments/text_compress.py
encoder_probe/experiments/small_vlm.py
encoder_probe/experiments/analyse.py
encoder_probe/experiments/run12_driver.py
encoder_probe/experiments/run_probe.py
encoder_probe/experiments/capacity_curve.py
experiments/readhook/smoke/run_smoke.py
experiments/readhook/vtchook/install.py
experiments/phase2_train.py
experiments/analyse_claude_qa.py
experiments/claude_qa_render.py
experiments/claude_qa_research.py
experiments/phase1_sweep.py
experiments/phase2_driver.py
experiments/phase1b_ocr.py
learned_glyphs/lg/report.py
learned_glyphs/experiments/calibrate_channel.py
scripts/job_queue.py
learned_glyphs/experiments/bigram_errors.py
scripts/check_claude_vision.py
learned_glyphs/experiments/transfer_eval.py
learned_glyphs/experiments/train_joint.py
scripts/make_readme_figures.py
scripts/build_data.py
"""),
    'ls_l_multi': ('listing', r"""encoder_probe/experiments:
total 372
drwxr-xr-x 2 root root  4096 Nov  7 10:03 __pycache__
-rw-r--r-- 1 root root  4629 Nov  3 20:53 analyse.py
-rw-r--r-- 1 root root  4383 Nov  4 12:58 analyse2.py
-rw-r--r-- 1 root root 14514 Nov  6 13:56 analyse_colour.py
-rw-r--r-- 1 root root  2542 Nov  5 08:35 analyse_g2lite.py
-rw-r--r-- 1 root root 13146 Nov  5 16:03 analyse_r3.py
-rw-r--r-- 1 root root 15568 Nov  7 10:03 analyse_run12.py
-rw-r--r-- 1 root root  4031 Nov  6 00:00 analyse_stage0.py
-rw-r--r-- 1 root root  3969 Nov  4 19:04 build_tokens2.py
-rw-r--r-- 1 root root 33734 Nov 10 13:03 capacity_curve.py
-rw-r--r-- 1 root root 17391 Nov  7 09:58 channel_capacity.py
-rw-r--r-- 1 root root 25085 Nov  7 07:03 finetune_qwen.py
-rw-r--r-- 1 root root  3892 Nov  5 06:35 g2lite_decode.py
-rw-r--r-- 1 root root 15231 Nov  4 16:44 longctx_bench.py
-rw-r--r-- 1 root root 13044 Nov  5 15:25 make_figures.py
-rw-r--r-- 1 root root 17952 Nov  4 19:42 quick_linear_readout.py
-rw-r--r-- 1 root root  6576 Nov  3 20:39 rgb_overlay_grid.txt
-rw-r--r-- 1 root root  9597 Nov 10 13:03 run12_driver.py
-rw-r--r-- 1 root root 24678 Nov  5 10:28 run_g2lite.py
-rw-r--r-- 1 root root  9500 Nov  4 12:52 run_probe.py
-rw-r--r-- 1 root root 13552 Nov  4 13:00 run_probe2.py
-rw-r--r-- 1 root root 28533 Nov  6 19:38 small_vlm.py
-rw-r--r-- 1 root root 22824 Nov  5 21:32 text_compress.py
-rw-r--r-- 1 root root  1082 Nov  3 22:03 wait_then_launch.sh
-rw-r--r-- 1 root root  9163 Nov  6 07:51 zeroshot_colour.py
-rw-r--r-- 1 root root  7974 Nov  7 06:58 zeroshot_info.py

vtc:
total 124
"""),
    'ls_la_root': ('listing', r"""total 120
drwxr-xr-x 14 root root  4096 Nov 11 03:35 .
drwxr-xr-x  3 root root  4096 Nov  2 20:29 ..
drwxr-xr-x  8 root root  4096 Nov 11 04:02 .git
-rw-r--r--  1 root root   959 Nov 11 03:35 .gitignore
drwxr-xr-x  3 root root  4096 Nov  2 20:40 .pytest_cache
-rw-r--r--  1 root root 25027 Nov 10 18:47 AGENTS.md
-rw-r--r--  1 root root 11031 Nov  5 15:21 PLAN.md
-rw-r--r--  1 root root 13803 Nov 10 21:26 README.md
drwxr-xr-x  3 root root  4096 Nov  2 20:47 assets
drwxr-xr-x  2 root root  4096 Nov  4 19:10 data
drwxr-xr-x  5 root root  4096 Nov 10 21:55 docs
drwxr-xr-x  9 root root  4096 Nov  5 15:25 encoder_probe
drwxr-xr-x  4 root root  4096 Nov 11 03:13 experiments
drwxr-xr-x  8 root root  4096 Nov  3 14:47 learned_glyphs
-rw-r--r--  1 root root   342 Nov  4 16:41 requirements.txt
drwxr-xr-x 14 root root  4096 Nov 11 03:30 results
drwxr-xr-x  3 root root  4096 Nov 10 21:46 scripts
drwxr-xr-x  3 root root  4096 Nov  2 21:45 tests
drwxr-xr-x  6 root root  4096 Nov  4 15:39 vtc
"""),
    'ls_plain': ('listing', r"""encoder_probe/experiments:
__pycache__
analyse.py
analyse2.py
analyse_colour.py
analyse_g2lite.py
analyse_r3.py
analyse_run12.py
analyse_stage0.py
build_tokens2.py
capacity_curve.py
channel_capacity.py
finetune_qwen.py
g2lite_decode.py
longctx_bench.py
make_figures.py
quick_linear_readout.py
rgb_overlay_grid.txt
run12_driver.py
run_g2lite.py
run_probe.py
run_probe2.py
small_vlm.py
text_compress.py
wait_then_launch.sh
zeroshot_colour.py
zeroshot_info.py

vtc/:
__init__.py
__pycache__
analysis.py
claude_eval.py
config.py
corpus.py
crnn
dictionary.py
layout.py
metrics.py
readers
schemes
"""),
    'ls_tree': ('listing', r""".
├── PLAN.md
├── RESULTS.md
├── RGB_OVERLAY.md
├── ROUND2.md
├── ROUND3.md
├── __init__.py
├── cache
│   ├── ckpt
│   │   ├── g2lite__g2lite__direct__rgb3_lg_pb_h5__3000__tp500__s0.lock
│   │   ├── g2lite__g2lite__frozen__latin_p8__3000__tp500__s0.lock
│   │   ├── g2lite__g2lite__frozen__rgb3_lg_pb_h5__3000__tp500__s0.lock
│   │   ├── g2lite__g2lite__trained__latin_p8__3000__tp500__s0.lock
│   │   ├── g2lite__g2lite__trained__latin_p8__3000__tp500__s0__mlr0.0001.lock
│   │   ├── g2lite__g2lite__trained__rgb3_lg_pb_h5__3000__tp500__s0.lock
│   │   ├── g2lite__g2lite__trained__rgb3_lg_pb_h5__3000__tp500__s0__mlr0.0001.lock
│   │   ├── g2lite__g2lite_blind__blind__latin_p16__1000__tp100__s0.lock
│   │   ├── g2lite__g2lite_blind__blind__rgb3_lg_pb_h5__1500__tp500__s0.lock
│   │   ├── g2lite__g2lite_diag__direct__latin_p8__500__tp500__s0.lock
│   │   ├── g2lite__g2lite_diag__frozen__latin_p16__1000__tp100__s0.lock
│   │   ├── g2lite__g2lite_diag__frozen__latin_p8__500__tp500__s0.lock
│   │   ├── g2lite__g2lite_diag__trained__latin_p8__500__tp500__s0.lock
│   │   ├── g2lite__g2lite_diag_mlr1e-4__trained__latin_p8__500__tp500__s0__mlr0.0001.lock
│   │   ├── g2lite__g2lite_ops__direct__rgb3_lg_pb_h5__3000__tp500__s0.lock
│   │   ├── g2lite__g2lite_ops__trained__latin_p8__3000__tp500__s0__mlr0.0001.lock
│   │   ├── g2lite__g2lite_ops__trained__rgb3_lg_pb_h5__3000__tp500__s0__mlr0.0001.lock
│   │   ├── old_collapsed
│   │   │   └── g2lite__g2lite__trained__rgb3_lg_pb_h5__3000__tp500__s0.pt
│   │   ├── s1__s1a__rgb1s_p16__1500.pt
│   │   └── s1__s1r__rgb1s_p16__1500.pt
│   ├── pages
│   │   ├── latin_p16_S448_tr100_te16.pkl
│   │   ├── latin_p5_S448_tr500_te16.pkl
│   │   ├── latin_p6_S448_tr500_te16.pkl
│   │   ├── latin_p8_S448_tr500_te16.pkl
│   │   ├── lg_pb_h5_S448_tr500_te16.pkl
│   │   └── rgb3_lg_pb_h5_S448_tr500_te16.pkl
│   ├── r3_adapters
│   │   └── colour_a1cpu__rgb2s_p16__500.pt
│   ├── s1_models
│   │   └── s1__rgb1s_p16__1500.pt
│   ├── tokens
│   │   ├── .vlm.lock
│   │   └── qwen25vl_h32
│   │       └── clean
│   │           ├── latin_p16_tr100_te16.pt
│   │           ├── latin_p8_tr500_te16.pt
│   │           └── rgb3_lg_pb_h5_tr500_te16.pt
│   └── weights
│       ├── deepseek_ocr_deepencoder.safetensors
│       └── qwen25vl3b_visual.safetensors
├── ep
│   ├── __init__.py
│   ├── deepencoder.py
│   ├── encoders.py
│   ├── flock.py
│   ├── g2lite.py
│   ├── geom.py
│   ├── hf_partial.py
│   ├── pages.py
│   ├── probe.py
│   ├── probe_grid.py
"""),
    'du_sh': ('listing', r"""28K	AGENTS.md
12K	PLAN.md
16K	README.md
17M	assets
422M	data
3.5M	docs
7.9G	encoder_probe
736K	experiments
1.5G	learned_glyphs
4.0K	requirements.txt
3.3G	results
284K	scripts
456K	tests
556K	vtc
"""),
    'ls_lR': ('listing', r"""experiments/readhook:
total 44
-rw-r--r-- 1 root root 4412 Nov 11 03:33 README.md
drwxr-xr-x 2 root root 4096 Nov 11 03:51 __pycache__
-rw-r--r-- 1 root root 1694 Nov 11 03:33 bash_output_hook.py
-rwxr-xr-x 1 root root 1580 Nov 11 03:13 hook.py
-rw-r--r-- 1 root root 1773 Nov 11 03:16 javamin.py
-rw-r--r-- 1 root root 2686 Nov 11 03:53 list.txt
-rw-r--r-- 1 root root 3202 Nov 11 03:22 pymin.py
-rw-r--r-- 1 root root 1899 Nov 11 03:13 render_code.py
drwxr-xr-x 4 root root 4096 Nov 11 03:31 smoke
drwxr-xr-x 5 root root 4096 Nov 11 03:53 vtchook

experiments/readhook/__pycache__:
total 12
-rw-r--r-- 1 root root 6614 Nov 11 03:51 pymin.cpython-313.pyc
-rw-r--r-- 1 root root 3425 Nov 11 03:32 render_code.cpython-313.pyc

experiments/readhook/smoke:
total 48
drwxr-xr-x 2 root root 4096 Nov 11 03:30 __pycache__
-rwxr-xr-x 1 root root  746 Nov 11 03:31 bash_guard_hook.py
-rw-r--r-- 1 root root 8220 Nov 11 03:29 checks.py
-rwxr-xr-x 1 root root 2418 Nov 11 03:29 img_edit_hook.py
-rwxr-xr-x 1 root root 3119 Nov 11 03:29 img_read_hook.py
-rw-r--r-- 1 root root  563 Nov 11 03:30 log_read_hook.py
-rw-r--r-- 1 root root 3202 Nov 11 03:29 pymin.py
-rw-r--r-- 1 root root 6378 Nov 11 03:31 run_smoke.py
drwxr-xr-x 8 root root 4096 Nov 11 03:27 tasks

experiments/readhook/smoke/__pycache__:
total 24
-rw-r--r-- 1 root root 16319 Nov 11 03:30 checks.cpython-313.pyc
-rw-r--r-- 1 root root  6620 Nov 11 03:29 pymin.cpython-313.pyc

experiments/readhook/smoke/tasks:
total 24
drwxr-xr-x 3 root root 4096 Nov 11 03:28 graph_hops
drwxr-xr-x 3 root root 4096 Nov 11 03:28 intervals
drwxr-xr-x 3 root root 4096 Nov 11 03:28 moving
drwxr-xr-x 3 root root 4096 Nov 11 03:28 orders
drwxr-xr-x 3 root root 4096 Nov 11 03:28 scheduler
drwxr-xr-x 3 root root 4096 Nov 11 03:28 text_stats

"""),
}
# fmt: on

# what detect() picks from the text alone (None: the hook relies on kind_for_command)
SNIFF = {"gitshow_s_merge": "gitlog", "ls_plain": None, "rg_nonum_init": None}


def flow_of(name):
    kind, text = SAMPLES[name]
    res = minify(text, hint=kind)
    assert res is not None, name
    return res[0]


def test_all_kinds_registered():
    for k in KINDS:
        assert k in REGISTRY and REGISTRY[k].cpt == O.CPT[k] and 1.3 < O.CPT[k] < 4
        assert k in O.LEGENDS
    assert {k for _, (k, _) in SAMPLES.items()} == set(KINDS)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_invariant_on_real_output(name):
    kind, text = SAMPLES[name]
    flow, m = minify(text, hint=kind)
    assert m.kind == kind and preserves(m, text, flow)
    assert "\n" not in flow and "\r" not in flow and "\x1b" not in flow
    assert "  " not in flow.replace("§", "").replace("»", "") or kind in ("pytest", "gitdiff")


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_sniff(name):
    kind, text = SAMPLES[name]
    m = detect(None, text)
    assert (m.kind if m else None) == SNIFF.get(name, kind)


@pytest.mark.parametrize("cmd,kind", [
    ("pytest -q", "pytest"), ("python3 -m pytest tests -x 2>&1 | tail -40", "pytest"),
    ("cd repo && uv run pytest -k slow", "pytest"), ("timeout 600 py.test", "pytest"),
    ("git diff", "gitdiff"), ("git --no-pager show HEAD~2", "gitdiff"), ("git log -p -3", "gitdiff"),
    ("diff -u a.py b.py", "gitdiff"), ("git -C repo log --oneline -20", "gitlog"), ("git log | grep fix", "gitlog"),
    ("git status", "gitstatus"), ("git status -s", "gitstatus"), ("git status && git diff", None),
    ("grep -rn foo . | head", "grep"), ("rg -n -C2 foo src", "grep"), ("git grep -n x", "grep"),
    ("cat x.log | grep -n ERROR", "grep"), ("find . | xargs grep -n TODO", "grep"),
    ("rg -l foo", "find"), ("find . -name '*.py' | sort", "find"), ("git ls-files", "find"),
    ("git diff --name-only", "find"), ("ls -la", "listing"), ("tree -L 2", "listing"), ("du -sh *", "listing"),
    ("cargo build", "diagnostics"), ("cargo test 2>&1", "log"), ("go vet ./...", "diagnostics"),
    ("go test ./...", "log"), ("npm test", "log"), ("npm run lint", "diagnostics"), ("npx tsc --noEmit", "diagnostics"),
    ("mypy src", "diagnostics"), ("ruff check .", "diagnostics"), ("ruff format .", None),
    ("gcc -Wall -c x.c", "diagnostics"), ("FOO=1 make -j4", "diagnostics"), ("javac Main.java", "diagnostics"),
    ("node --test", "log"), ("docker logs web", "log"), ("python3 x.py", None), ("echo hi", None),
    ("pytest | wc -l", None), ("", None),
])
def test_kind_for_command(cmd, kind):
    assert O.kind_for_command(cmd) == kind


# ------------------------------------------------------------------------------------------- caret encoding
CARET_LINE = re.compile(r"^(\s*\|)?\s*[~^]+\s*$")


def decode_groups(flow):
    """Independent decoder of the bracket encoding: [(marked text, caret run)] in flow order."""
    out = []
    for m in re.finditer(r"⟨([^⟩]*)⟩|⟦((?:[^⟦⟧]|⟦[^⟦⟧]*⟧)*)⟧", flow):
        if m.group(1) is not None:
            out.append((m.group(1), "~" * len(m.group(1))))
            continue
        inner = m.group(2)
        if "⟦" not in inner:
            out.append((inner, "^" * len(inner)))
            continue
        plain, run = "", ""
        for part in re.split(r"(⟦[^⟦⟧]*⟧)", inner):
            if part.startswith("⟦"):
                plain, run = plain + part[1:-1], run + "^" * (len(part) - 2)
            else:
                plain, run = plain + part, run + "~" * len(part)
        out.append((plain, run))
    return out


@pytest.mark.parametrize("name", ["pytb_zero", "pytb_chain", "pytb_deep", "pytb_syntax", "pytest_", "diag_gcc",
                                  "diag_mypy_pretty", "diag_tsc_pretty", "diag_ruff", "diag_javac2", "diag_clang"])
def test_caret_spans_match_the_columns(name):
    kind, text = SAMPLES[name]
    lines = O.norm_term(text).split("\n")         # not the separator collapse: carets keep their width
    want = []
    for src, car in zip(lines, lines[1:], strict=False):
        if not CARET_LINE.match(car) or "\t" in src + car:
            continue
        body = car.split("|", 1)[1] if car.lstrip().startswith("|") else car
        off = len(car) - len(body)
        a = off + len(body) - len(body.lstrip())
        run = body.strip()
        if len(src.rstrip()) >= a + len(run) and re.fullmatch(r"~*\^+~*|~+", run) and src[a:a + len(run)].strip():
            want.append((src[a:a + len(run)], run))
    got = decode_groups(flow_of(name))
    assert want and got == want


def test_caret_examples():
    assert '⟦item["value"] * scale ⟦/⟧ item["weight"]⟧' in flow_of("pytb_zero")
    assert "»»f.write(⟦encode⟦(items)⟧⟧)" in flow_of("pytb_deep")
    assert "12 | int n = ⟦⟦u⟧ndeclared_var⟧ + 1; |" in flow_of("diag_gcc")        # gutter | kept after the line
    f = flow_of("diag_cargo_build")
    assert "»»19 | println!(\"{}\", s ⟦+⟧ m.get(\"k\")); | »»»no implementation for `i32 + Option<&i32>`" in f
    t = O.caret_expand("§1 x = ⟦a ⟦+⟧ b⟧ §2 y")
    assert t.split() == "§1 x = a + b ~~^~~ §2 y".split()


def test_caret_fallbacks_keep_the_line():
    assert "»»pass »»^ §4 TabError" in flow_of("pytb_tab")                          # tab in the source line
    f = flow_of("diag_gcc")
    assert "»| ~^ ~~~~~~~~~~~" in f                                                  # two runs: kept as a line
    assert "»7 | return p->x * p->y * scale »| ^" in f                               # caret past the line end


# ----------------------------------------------------------------------------------------------- pytest
def test_pytest_structure():
    f = flow_of("pytest_")
    assert "§1 === test session starts ===" in f and "=== FAILURES ===" in f
    assert "___ TestPaths.test_join ___" in f and "___ test_compute_param[2-7] ___" in f
    assert "FAILED sub/test_more.py::TestPaths::test_join - AssertionError: assert 'a/b' ..." in f
    assert "»def test_compute(): »»x = compute(3) »»> assert x == 10 §76 E assert 9 == 10" in f
    assert "E »+ hello  world, this is a test" in f                    # E lines keep their inner whitespace
    assert "_ _ _ §" in f                                              # frame separator collapsed


def test_pytest_colour_and_separators_normalised():
    kind, text = SAMPLES["pytest_color"]
    assert "\x1b[" in text and "=" * 20 in text
    n = REGISTRY[kind].normalise(text)
    assert "\x1b" not in n and "=" * 4 not in n.replace("== ", "")
    assert flow_of("pytest_color").count("===") >= 2


# ----------------------------------------------------------------------------------------------- git diff
OLD = """import os


class Store:
    def __init__(self, path):
        self.path = path
        self.items = {}

    def get(self, key):
        return self.items[key]

    def put(self, key, value):
        self.items[key] = value


def main():
    s = Store("db")
    s.put("a", 1)
    print(s.get("a"))
""" + "".join(f"# filler line {i}\n" for i in range(30)) + """
def tail():
    return 0
"""
NEW = (OLD.replace("        return self.items[key]\n", "        if key not in self.items:\n            raise KeyError(key)\n"
                   "        return self.items[key]\n")
          .replace("import os\n", "import os\nimport sys\n").replace("    return 0\n", "    return 1\n")
          .replace("# filler line 7\n", ""))


def test_gitdiff_tags_are_file_line_numbers():
    diff = "diff --git a/s.py b/s.py\nindex 1..2 100644\n" + "".join(
        difflib.unified_diff(OLD.splitlines(True), NEW.splitlines(True), "a/s.py", "b/s.py", n=2))
    flow, m = minify(diff, hint="gitdiff")
    old, new = OLD.split("\n"), NEW.split("\n")
    hits = re.findall(r"§(\d+) ([+-]?)(»*)(.*?)(?= §\d+ | @@ |$)", flow)
    assert len(hits) == sum(1 for x in diff.split("\n")[4:] if x[:1] in "+- " and x.strip() and not x.startswith("@@"))
    for n, mk, lv, code in hits:
        src = (old if mk == "-" else new)[int(n) - 1]
        assert src.strip() == code
        assert len(lv) == (len(src) - len(src.lstrip())) // 4
    assert "@@ -1,3 +1,4 @@ §1 import os §2 +import sys" in flow and "diff --git a/s.py b/s.py index 1..2 100644 --- a/s.py +++ b/s.py" in flow


def test_gitdiff_real_samples():
    f = flow_of("gitdiff_head")
    assert "§8 +»»self.cache = {}" in f and "§15 -»»return self.data[\"settings\"][key]" in f
    assert "§24 +»total  = 0.0" in f                                   # inner whitespace kept (a whitespace edit)
    assert "\\ No newline at end of file" in f and "Binary files /dev/null and b/blob.bin differ" in f
    g = flow_of("gitshow_s")
    assert g.startswith("commit ") and "»Config cache; float totals; main takes argv" in g
    assert "--- a/src/app.py +++ b/src/app.py @@ -5,6 +5,7 @@ class Config:" in g


def test_gitdiff_removed_header_lookalike():
    d = "@@ -1,3 +1,2 @@\n a\n--- b\n c\n"                            # a removed line "-- b", not a header
    flow, _ = minify(d, hint="gitdiff")
    assert flow == "@@ -1,3 +1,2 @@ §1 a §2 --- b §2 c"


# -------------------------------------------------------------------------------------------- log/status
def test_gitlog_one_tag_per_entry():
    f = flow_of("gitlog_s_default")
    assert f.count("§") == SAMPLES["gitlog_s_default"][1].count("\ncommit ") + 1
    assert re.search(r"§1 commit [0-9a-f]{40} Author: \S+ <\S+> Date: .*? »notes: entry 3 §", f)
    o = flow_of("gitlog_s_oneline")
    assert o.count("§") == len(SAMPLES["gitlog_s_oneline"][1].strip().split("\n"))


def test_gitstatus_short_codes_roundtrip():
    kind, text = SAMPLES["gitstatus_short"]
    flow, m = minify(text, hint=kind)
    assert "§4 .M src/a.py" in flow and "§2 M. docs/readme.md" in flow and "§6 MM src/pkg/c.py" in flow
    back = m.expand(flow)
    assert re.sub(r"[ \t]+", " ", back).split() == text.split()
    assert " M src/a.py" in back and "M  docs/readme.md" in back and "MM src/pkg/c.py" in back
    f = flow_of("gitstatus_long")
    assert "§4 renamed: old.txt -> config.cfg" in f and "»(use \"git restore --staged <file>...\" to unstage)" in f


# ------------------------------------------------------------------------------------------------- grep/find
@pytest.mark.parametrize("name", ["grep_rn_render", "grep_ctx_scheme", "rg_C2_fitem", "grep_n_single", "rg_column",
                                  "rg_C1_subprocess", "rg_nonum_init"])
def test_grep_expand_restores_lines(name):
    kind, text = SAMPLES[name]
    flow, m = minify(text, hint=kind)
    want = [nonws(x) for x in text.split("\n") if x.strip()]
    got = [nonws(x) for x in m.expand(flow).split("\n") if x.strip()]
    assert got == want


def test_grep_layout():
    f = flow_of("grep_ctx_scheme")
    assert f.startswith("»vtc/schemes/substitution.py §123- §124 class SubstitutionScheme(Scheme): §125- def")
    assert " »-- »vtc/schemes/base.py §17- §18 class Scheme(ABC):" in f
    g = flow_of("grep_n_single")
    assert g.startswith("»§") and "»" not in g[1:]                      # one group, no path


@pytest.mark.parametrize("name", ["find_py", "find_exp", "rg_l"])
def test_find_expand_restores_paths(name):
    kind, text = SAMPLES[name]
    flow, m = minify(text, hint=kind)
    assert m.expand(flow).split("\n") == [x.strip() for x in text.split("\n") if x.strip()]
    assert len(flow) < 0.7 * len(text)


def test_find_with_spaces_is_not_grouped():
    t = "./a b/c.txt\n./a b/d.txt\n./e.txt\n"
    flow, m = minify(t, hint="find")
    assert flow == "§1 ./a b/c.txt §2 ./a b/d.txt §3 ./e.txt"


def test_listing_layout():
    t = flow_of("ls_tree")
    assert t.startswith("§1 . ├── PLAN.md ├── RESULTS.md") and " │├── ckpt ││├── " in t
    tree = ".\n├── a\n│   ├── b.py\n│   └── c\n│       └── d.py\n└── e\n    └── f.txt\n\n3 directories, 3 files\n"
    flow, _ = minify(tree, hint="listing")
    assert flow == "§1 . ├── a │├── b.py │└── c │»└── d.py └── e »└── f.txt §9 3 directories, 3 files"
    ll = flow_of("ls_l_multi")
    assert re.search(r"§2 total \d+ drwxr-xr-x \d+ root root \d+ \w+ +\d+ [\d:]+ __pycache__ -rw-r--r--", ll)
    p = flow_of("ls_plain")
    assert p.count("§") == 2 and "§1 encoder_probe/experiments: __pycache__ analyse.py" in p


# -------------------------------------------------------------------------------------------- logs, generic
def test_log_progress_bars_and_tags():
    kind, text = SAMPLES["log_a1cpu"]
    assert "\r" in text
    f = flow_of("log_a1cpu")
    assert "Loading weights: 27%" not in f and "824/824" in f
    g = flow_of("log_pylogging")
    assert re.match(r"§1 2026-11-11 \d\d:\d\d:\d\d,\d{3} INFO \[worker.pool\] processed batch 0", g)
    assert "Traceback (most recent call last): »File" in g


def test_generic_policy_tags_and_levels():
    t = "top line\n  child\n    grandchild\n  child2\nnext top\n\tat java.Foo(Foo.java:3)\n"
    flow, _ = minify(t, hint="log")
    assert flow == "§1 top line »child »»grandchild »child2 §5 next top »at java.Foo(Foo.java:3)"


# ----------------------------------------------------------------------------------------------- edge cases
def test_refusals():
    assert minify("a § b\n" * 50, hint="log") is None                 # marker characters: text stays text
    assert minify("x ⟦y⟧\n" * 50, hint="traceback") is None            # caret brackets in the text
    assert minify("x ⟦y⟧:1:z\n", hint="grep") is not None
    assert minify("", hint="pytest") is None
    assert minify("   \n\n", hint="log") is None


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_handles_every_sample(kind):
    """A wrong hint (the hook trusts kind_for_command) costs density, never correctness."""
    ok = 0
    for name, (_, text) in SAMPLES.items():
        res = minify(text, hint=kind)
        if res is not None:
            assert preserves(REGISTRY[kind], text, res[0]), (kind, name)
            ok += 1
    assert ok >= len(SAMPLES) - 2


def test_garbage_and_odd_input():
    odd = ["\x00\x01binary\xff\n" * 20, "\t\t\n" * 10 + "x", "^^^^\n~~~~\n", "  |  ^\n", "@@ -a +b @@\n+x\n",
           "a:1:\n:2:b\n--\n", "-" * 500, "x\r" * 50 + "\n", "\x1b[31m\x1b[0m\n" * 5 + "ok"]
    for kind in KINDS:
        for t in odd:
            res = minify(t, hint=kind)
            assert res is None or preserves(REGISTRY[kind], t, res[0])


def test_large_output_is_fast():
    big = (SAMPLES["pytest_"][1] * 40)[:400_000]
    t0 = time.time()
    for kind in ("pytest", "log", "grep", "gitdiff"):
        assert minify(big, hint=kind) is not None
    assert time.time() - t0 < 8


def test_tag_modes_for_measurement():
    text = SAMPLES["pytest_"][1]
    try:
        O.TAG_MODE = "all"
        a = O.pytest_flow(text)
        O.TAG_MODE = "none"
        b = O.pytest_flow(text)
    finally:
        O.TAG_MODE = "default"
    d = O.pytest_flow(text)
    assert len(b) < len(d) < len(a) and "§" not in b


# ---------------------------------------------------------------------------------------------- the hook
def test_bash_hook_renders_pytest_output(tmp_path):
    kind, text = SAMPLES["pytest_"]
    ev = {"tool_name": "Bash", "tool_input": {"command": "python3 -m pytest -q 2>&1"},
          "tool_response": {"stdout": text, "stderr": "", "interrupted": False, "isImage": False}}
    env = dict(os.environ, VTCHOOK_CACHE=str(tmp_path / "cache"), VTCHOOK_LOG=str(tmp_path / "log.jsonl"),
               VTCHOOK_BUDGET="88.2")
    p = subprocess.run([sys.executable, os.path.join(HERE, "bash_hook.py")], input=json.dumps(ev),
                       capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)["hookSpecificOutput"]
    assert out["updatedToolOutput"]["isImage"] and "(pytest)" in out["additionalContext"]
    rec = json.loads(open(tmp_path / "log.jsonl").read().strip().split("\n")[-1])
    assert rec["kind"] == "pytest" and rec["image_tokens"] * 3 < rec["text_tokens"]
