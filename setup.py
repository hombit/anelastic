from setuptools import setup
from pip.req import parse_requirements

setup(
    name='anelastic',
    version='',
    url='',
    license='',
    author='Konstantin Malanchev',
    author_email='',
    description='',
    packages=['polynomials'],
    install_reqs=parse_requirements('requirements.txt', session='hack'),
    scripts=['bin/modal_analysis'],
    test_suite='polynomials.test.suite',
)
