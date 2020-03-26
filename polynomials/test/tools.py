import unittest
from doctest import DocTestSuite
from importlib import import_module


class DocTestLoader(unittest.TestLoader):
    def loadTestsFromModule(self, module, *args, pattern=None, **kws):
        return DocTestSuite(module)


defaultDocTestLoader = DocTestLoader()
