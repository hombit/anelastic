__all__ = ('test_polynomcoeffs', 'test_background')


def __doctest_suite():
    from . import tools
    from os.path import dirname
    parent_path = dirname(__path__[0])
    test_suite = tools.defaultDocTestLoader.discover(start_dir=parent_path,
                                                     pattern='[a-zA-Z]*.py')
    return test_suite


def __discoverable_test_suite():
    from unittest import defaultTestLoader
    test_suite = defaultTestLoader.discover(start_dir='.',
                                            pattern='test_*.py')
    return test_suite


def __test_suite_from_suite_functions():
    from importlib import import_module
    from unittest import TestSuite
    test_suite = TestSuite()
    for module_name in __all__:
        try:
            module = import_module('.' + module_name, __package__)
            module_suite = module.suite()
            test_suite.addTest(module_suite)
        except AttributeError:
            pass
    return test_suite


def suite():
    from unittest import TestSuite
    test_suite = TestSuite()
    test_suite.addTests((
        __doctest_suite(),
        __discoverable_test_suite(),
        __test_suite_from_suite_functions(),
    ))
    return test_suite
