import os

_TEST_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_TEST_ROOT, "..", "src"))
# if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)
_PATH_DATA = os.path.abspath(os.path.join(_TEST_ROOT, "..", "data", "splits"))
