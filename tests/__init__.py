import os
import sys


_TEST_ROOT = os.path.dirname(__file__)
print(_TEST_ROOT)
_PROJECT_ROOT = os.path.abspath(os.path.join(_TEST_ROOT, '..', 'src'))
#if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)
_PATH_DATA = os.path.abspath(os.path.join(_TEST_ROOT, '..', 'data', 'splits'))