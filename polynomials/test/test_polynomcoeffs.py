import copy
import sympy, sympy.abc
from sympy.abc import xi
import numpy as np
import unittest

from polynomials import DifEq


class BasicTestDifEq():
    """
    This is a basic class. Subclasses of this class can be used for test DifEq
    accuracy.

    Parameters
    ----------
    k_max : int, optional
        Will be used as one of the arguments of DifEq. Default is 6
    harmonic : int, optional
        Number that characterise eigenvalue of the problem. Default is 0

    Attributes
    ----------
    parameters_for_DifEq : tuple
        Parameters to initialise DifEq object: DifEq(*parameters_for_DifEq)
    de : DifEq
        DifEq object. Will be created at the first lookup
    ev : float or complex
        Eigenvalue of the problem
    ef : sympy expression
        Eigenfunction of the problem
    cs : numpy.ndarray
        1-D array with coefficients of polynomial representation of
        eigenfunction

    Methods
    -------
    dev()
        Difference between analytical and DifEq eigenvalues
    dcs()
        Difference between analytical and DifEq eigenfunction

    """
    _de = None
    _cs = None
    _calibr_k = None
    _default_k_max = 6
    # Ask TestSuite to skip dcs test if it can be very long
    _skip_dcs_test = False

    def __init__(self, k_max=None, harmonic=0):
        if k_max is None:
            self.k_max = self._default_k_max
        else:
            self.k_max = k_max
        self.harmonic = harmonic

    def _produce_cs(self):
        self._cs = self._ef\
            .series( x=xi, n=self.k_max+1 )\
            .removeO()\
            .evalf()\
            .as_poly()\
            .all_coeffs()
        self._cs.reverse()
        self._cs = np.array(self._cs)
        # Slow, but we have right length and right dtype
        if self._cs.size < self.k_max + 1:
            self._cs = np.append(self._cs, [0] * (self.k_max + 1 - self._cs.size))
        # Assume that one of first two polynomial coefficients is non-zero
        self._calibr_k = np.argmax(np.abs(self._cs[:2]))
        self._cs /= self._cs[self._calibr_k]
        self._cs.flags.writeable = False

    def dev(self):
        """
        Difference between analytical and DifEq eigenvalues.

        Returns
        -------
        float
            Relative error of DifEq eigenvalue modulus

        """
        de_ev = self.de.findEigenValue(self.ev)
        return abs(self.ev - de_ev) / abs(self.ev)

    def dcs(self):
        """
        Difference between analytical and DifEq eigenfunctions.

        Returns
        -------
        numpy.ndarray
            1-D array of ratio of analytical and DifEq coefficients of
            polynomial representation of eigenfunction (cs)

        """
        de_ev = self.de.findEigenValue(self.ev)
        de_cs = self.de.computeEigenFunctions(
            f_symbol = self._f,
            calibr_k = self.calibr_k
        )[self._f]
        return np.asarray(self.cs - de_cs, dtype=type(de_ev))

    @property
    def parameters_for_DifEq(self):
        return copy.deepcopy( self._parameters_for_DifEq )

    @property
    def ev(self):
        return self._ev

    @property
    def ef(self):
        return copy.copy( self._ef )

    @property
    def cs(self):
        if self._cs is None:
            self._produce_cs()
        return self._cs

    @property
    def calibr_k(self):
        if self._calibr_k is None:
            self._produce_cs()
        return self._calibr_k

    @property
    def de(self):
        if self._de is None:
            self._de = DifEq( *self._parameters_for_DifEq )
        return self._de


class HarmonicOsc(BasicTestDifEq):
    """
    Test suite for DifEq for classical harmonic oscillator problem
    f''(xi) +- omega^2 f(xi) = 0, f'(xi=0) = 0, f(xi=1) = 0.
    The solution of this problem is cos( |omega| xi ).
    
    See full documentation of the class in basicTestDifEq.

    Parameters
    ----------
    wrongsign : bool
        If False than use classical form of the problem f'' + omega^2 f = 0.
        If True use f'' - omega^2 f = 0, which eigenvalues for omega are
        imaginaries. Default is False

    """

    def __init__(self, wrongsign=False, **kwargs):
        super().__init__(**kwargs)
        
        self._sign = (-1)**int(wrongsign)
        omega = sympy.abc.omega
        self._f = sympy.Function('f')
        self._fs = np.array( [self._f], dtype=sympy.Function )
        self._eqs = [
            (
                self._sign * omega**2 * self._f(xi) +\
                    + sympy.diff(self._f(xi),xi,2),
                2
            )
        ]
        self._boundconds = [
            (
                sympy.diff(self._f(xi),xi,1),
                0.
            ),
            (
                self._f(xi),
                1.
            )
        ]
        
        self._parameters_for_DifEq = (
            self._fs,
            self._eqs,
            self._boundconds,
            omega,
            self.k_max,
        )

        self._sqrtsign = 1.j if self._sign == -1 else 1.
        self._ev = self._sqrtsign * (np.pi/2 + np.pi * self.harmonic)
        self._ef = sympy.cos( self._ev / self._sqrtsign * xi )


class BoundedHydrogen(BasicTestDifEq):
    """
    Test suite for DifEq for the problem
    f''(xi) + omega^2 f(xi) / (1-xi) = 0,
    f(xi=0) = 0,
    f(xi=1) = 0.
    This problem is equivalent for the following problem:
    f''(x) + omega^2 f(x) / x = 0,
    f(x=0) = 0,
    f(x=1) = 0,
    x = 1-xi.
    The solution if the last problem is
    f(x) = sqrt(x) omega J_1(2 sqrt(x) omega),
    where J_1 is the Bessel function of the first kind.
    
    See full documentation of the class in basicTestDifEq.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        omega = sympy.abc.omega
        self._f = sympy.Function('f')
        self._fs = np.array( [self._f], dtype=sympy.Function )
        self._eqs = [
            (
                omega**2 * self._f(xi) / (1-xi) +\
                    + sympy.diff(self._f(xi),xi,2),
                2
            )
        ]
        self._boundconds = [
            (
                self._f(xi),
                0.
            ),
            (
                self._f(xi),
                1.
            )
        ]
        
        self._parameters_for_DifEq = (
            self._fs,
            self._eqs,
            self._boundconds,
            omega,
            self.k_max,
        )

        import scipy.special
        self._ev = scipy.special.jn_zeros( 1, self.harmonic+1 )[-1] / 2

        y = sympy.sqrt(1-xi) * self._ev
        self._ef = y * sympy.special.bessel.besselj(1, 2*y)
        

class OneAndHalf(BasicTestDifEq):
    """
    Test suite for DifEq for the problem
    f''(xi) + omega^2 f(xi) / (1-xi)^(3/2) = 0,
    f(xi=0) = 0,
    f(xi=1) = 0.
    This problem is equivalent for the following problem:
    f''(x) + omega^2 f(x) / x^(3/2) = 0,
    f(x=0) = 0,
    f(x=1) = 0,
    x = 1-xi.
    The solution if the last problem is
    f(x) = sqrt(1-x) omega^2 J_2(4 x^(1/4) omega),
    where J_2 is the Bessel function of the first kind.
    
    See full documentation of the class in basicTestDifEq.

    """
    _default_k_max = 22

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        omega = sympy.abc.omega
        self._f = sympy.Function('f')
        self._fs = np.array( [self._f], dtype=sympy.Function )
        self._eqs = [
            (
                omega**2 * self._f(xi) / (1-xi)**(1.5) +\
                    + sympy.diff(self._f(xi),xi,2),
                2
            )
        ]
        self._boundconds = [
            (
                self._f(xi),
                0.
            ),
            (
                self._f(xi),
                1.
            )
        ]
        
        self._parameters_for_DifEq = (
            self._fs,
            self._eqs,
            self._boundconds,
            omega,
            self.k_max,
        )

        import scipy.special
        self._ev = scipy.special.jn_zeros( 2, self.harmonic+1 )[-1] / 4

        y = (1-xi)**(1/4) * self._ev
        self._ef = y**2 * sympy.special.bessel.besselj(2, 4*y)

    @property
    def _skip_dcs_test(self):
        return self.k_max > 15


################################################################################
################################################################################


class BaseTestCase():
    decimal = 1
    _object = None

    def test_dev(self):
        'Check eigenvalues for {}'.format(self.class_to_test.__name__)
        np.testing.assert_almost_equal(self.object.dev(),
                                       0.,
                                       decimal=self.decimal)

    def test_dcs(self):
        'Check polynomial coefficients for {}'.format(self.class_to_test.__name__)
        if self.object._skip_dcs_test:
            raise unittest.SkipTest('Skipping very long test')
        np.testing.assert_almost_equal(self.object.dcs(),
                                       0.,
                                       decimal=self.decimal)
    @property
    def object(self):
        if self._object is None:
            self._object = self.class_to_test()
        return self._object

    @property
    def class_to_test(self):
        raise NotImplemented('You should specify class_to_test')


def suite():
    test_suite = unittest.TestSuite()
    for class_to_test in BasicTestDifEq.__subclasses__():
        name = class_to_test.__name__ + 'TestCase'
        test_case_class = type(name,
                         (BaseTestCase, unittest.TestCase,),
                         {'class_to_test': class_to_test})
        tests_from_case = unittest.defaultTestLoader.loadTestsFromTestCase(test_case_class)
        test_suite.addTest(tests_from_case)
    return test_suite
