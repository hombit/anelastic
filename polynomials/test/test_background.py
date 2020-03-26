import unittest
from sympy import diff
from sympy import lambdify

from polynomials.background import *


class BackgroundStructureSymbolicTestCase(unittest.TestCase):
    def dxdT_eq_diffx(self, n):
        'Number of various substitutions is n**3'
        bckgrnd = BackgroundStructureSymbolic()
        x = bckgrnd.x
        dxdT = bckgrnd.dxdtheta
        ratio = diff(x ,bckgrnd.theta) / dxdT
        for a in np.linspace(0., 3., n):
            for b in np.linspace(0., 3., n):
                for T in np.linspace(0.1, 0.9, n):
                    with self.subTest(a=a, b=b, T=T):
                        np.testing.assert_almost_equal(
                            ratio.subs(bckgrnd.a, a).subs(bckgrnd.b, b).subs(bckgrnd.theta, T).evalf(),
                            1.
                        )

    def test_dxdT_equals_diffx(self):
        'Test that derivative of BackgroundStructure.x is BackgroundStructure.dxdT'
        # Run n**3 tests for various values of a, b and T
        self.dxdT_eq_diffx(n=4)

    def test_polytrope(self):
        'Test that for a = b = 0 background structure is polytropic'
        a = b = 0
        B = 2
        bckgrnd = BackgroundStructureSymbolic()

        x = bckgrnd.x_subs(a=a, b=b, B=B)
        self.assertEqual( x.simplify(), sqrt(1 - bckgrnd.theta) )

        dxdT = bckgrnd.dxdtheta_subs(a=a, b=b, B=B)
        self.assertEqual( dxdT.simplify(), - 1 /( 2 * sqrt(1 - bckgrnd.theta) ) )


class BackgroundStructureExpressionsTestCase(unittest.TestCase):
    def equvivalent_with_BackgroundStructureSymbol(self, n):
        'Number of various substitutions is n**3'
        Pr = 0.052
        cP = 2.5
        dlogOmegadlogR = -1.5
        theta = abc.theta

        bg_symbolic = BackgroundStructureSymbolic(theta=theta)

        for a in np.linspace(0., 3., n):
            for b in np.linspace(0., 3., n):
                gas = GasProperties( a = a, b = b, Pr = Pr, cP = cP )
                bckgrnd = BackgroundStructure(gas, dlogOmegadlogR=dlogOmegadlogR, theta=theta)
                B = bckgrnd.B
                ratio = bg_symbolic.x_subs(a=a, b=b, B=B) / bckgrnd.x
                thetabound = bckgrnd.thetabound
                Ts = thetabound + (1 - thetabound) * np.linspace(0.1, 0.9, n)
                for T in Ts:
                    with self.subTest(a=a, b=b, T=T):
                        np.testing.assert_almost_equal(
                            ratio.subs(theta, T).evalf(),
                            1.
                        )

    def test_equvivalent_with_BackgroundStructureSymbol(self):
        'Compare x and dx/dtheta obtained from BackgroundStructure and BackgroundStructureSymbolic'
        # Run n**3 tests for various values of a, b and T
        self.equvivalent_with_BackgroundStructureSymbol(n=4)



class BackgroundStructureBounderiesTestCase(unittest.TestCase):
    def test_neutral(self):
        'Test exception when set supercritical Prandtl number'
        gas = GasProperties(
            a = 0.5, b = 0.5,
            Pr = 2./ 3.,
            cP=2.5,
        )
        dlogOmegadlogR = -1.5

        with self.assertRaises(ValueError):
            bckgrnd = BackgroundStructure(gas, dlogOmegadlogR=dlogOmegadlogR)

    def test_polytrope(self):
        'Test case of polytrope structure when whole disk is laminar'
        dlogOmegadlogR = -1.5
        gas = GasProperties(
            a=0, b=0,
            Pr=0.5 / dlogOmegadlogR ** 2,  # Pr is subcritical,
            cP=2.5,
        )

        xi_expected = np.sqrt(2. / (gas.Pr / gas.cV * dlogOmegadlogR ** 2))

        bckgrnd = BackgroundStructure(gas, dlogOmegadlogR=dlogOmegadlogR)

        with self.subTest('bounds'):
            np.testing.assert_array_almost_equal([bckgrnd.xbound, bckgrnd.thetabound], [1., 0.])

        with self.subTest('xi'):
            np.testing.assert_almost_equal(bckgrnd.xi, xi_expected)

    def test_ion(self):
        'Test boundaries for the case of fully ionized gas'
        gas = GasProperties(
            a=2.5, b=2.5,
            Pr=0.052,
            cP=2.5,
        )
        dlogOmegadlogR = -1.5

        xbound_expected, Tbound_expected = 0.938, 0.372
        decimal = 3

        bckgrnd = BackgroundStructure(gas, dlogOmegadlogR=dlogOmegadlogR)

        np.testing.assert_array_almost_equal(
            [bckgrnd.xbound, bckgrnd.thetabound],
            [xbound_expected, Tbound_expected],
            decimal=decimal
        )


class BackgroundStructureDensityPressureTestCase(unittest.TestCase):
    def polytrope(self, n):
        'n is a number of theta points to test'
        dlogOmegadlogR = -1.5
        gas = GasProperties(
            a=0, b=0,
            Pr=0.5 / dlogOmegadlogR ** 2,  # Pr is subcritical,
            cP=2.5,
        )

        bckgrnd = BackgroundStructure(gas, dlogOmegadlogR=dlogOmegadlogR)

        expected_polyindex = bckgrnd.xi ** 2 * bckgrnd.gamma / 2. - 1

        lmbd = bckgrnd.get_lmbd()
        dlnlmbddlntheta = diff(ln(lmbd), bckgrnd.theta) * bckgrnd.theta
        polyindex = np.empty(n, dtype=float)
        for i, T in enumerate(np.linspace(0.1, 0.9, n)):
            polyindex[i] = dlnlmbddlntheta.subs(bckgrnd.theta, T).evalf()
        np.testing.assert_array_almost_equal(polyindex, expected_polyindex)

    def test_polytrope(self):
        'Check polytrope index'
        n = 10
        self.polytrope(n=n)

    def dlnlmbddT_eq_difflnlmbd(self, n):
        'Number of various substitutions is n**3'
        Pr = 0.052
        cP = 2.5
        dlogOmegadlogR = -1.5

        theta = abc.theta

        for a in np.linspace(0., 2.5, n):
            for b in np.linspace(0., 2.5, n):
                gas = GasProperties(a=a, b=b, Pr=Pr, cP=cP)
                bckgrnd = BackgroundStructure(gas, dlogOmegadlogR=dlogOmegadlogR, theta=theta)
                lmbd = bckgrnd.get_lmbd()
                dlnlmbddtheta = lambdify(theta, bckgrnd.dlnlmbddtheta)
                lmbd_quad = lambda T: np.exp(scipy.integrate.quad(dlnlmbddtheta, 1., T)[0])
                thetabound = bckgrnd.thetabound
                Ts = thetabound + (1 - thetabound) * np.linspace(0.1, 0.9, n)
                for T in Ts:
                    with self.subTest(a=a, b=b, T=T):
                        np.testing.assert_almost_equal(
                            lmbd.subs(theta, T).evalf() / lmbd_quad(T),
                            1.
                        )

    def test_density(self):
        'Check that derivative of obtained density distribution equals analytical derivative'
        n = 2
        self.dlnlmbddT_eq_difflnlmbd(n=n)
