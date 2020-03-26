#!/usr/bin/env python3


import numpy as np
import scipy.optimize, scipy.integrate
import sympy
from copy import copy, deepcopy
from functools import lru_cache
from sympy import abc
from sympy import pi
from sympy import hyper, gamma, sqrt, exp, ln
from sympy import Rational


class GasProperties(object):
    def __init__(self, a, b, Pr, cP=Rational(5,2)):
        self.a = a
        self.b = b
        self.Pr = Pr
        self.cP = cP
        self.cV = self.cP - 1
        self.gamma = self.cP / self.cV  


def _hyp_2F1_for_x(a, b, z):
    return hyper(
        ( Rational(1, 2), (a + 1)/(a + b + 1), ),
        ( (2*a + b + 2)/(a + b + 1), ),
        z
    )


class BackgroundStructureSymbolic(object):
    def __init__(self, a=abc.a, b=abc.b, B=abc.B, theta=abc.theta):
        self.__a = a
        self.__b = b
        self.__B = B
        self.__theta = theta

        self.__2F1_1 = self._2F1(1.)
        self.__coef = sqrt( (self.__a+self.__b+1)/( 2 * self.__B ) )

        self.__x = self.__coef  / (self.__a+1) * ( self.__2F1_1 - self.__theta**(self.__a+1) * self._2F1(self.__theta**(self.__a+self.__b+1)) )
        self.__dxdtheta = -self.__coef * self.__theta**self.__a / sqrt( 1 - self.__theta**(self.__a+self.__b+1) )
    
    def _2F1(self, z):
        return _hyp_2F1_for_x(self.__a, self.__b, z)

    def x_subs(self, a=None, b=None, B=None, theta=None):
        x = deepcopy(self.__x)
        if a is not None:
            x = x.subs(self.__a, a)
        if b is not None:
            x = x.subs(self.__b, b)
        if B is not None:
            x = x.subs(self.__B, B)
        if theta is not None:
            x = x.subs(self.__theta, theta)
        return x

    def dxdtheta_subs(self, a=None, b=None, B=None, theta=None):
        dxdtheta = deepcopy(self.__dxdtheta)
        if a is not None:
            dxdtheta = dxdtheta.subs(self.__a, a)
        if b is not None:
            dxdtheta = dxdtheta.subs(self.__b, b)
        if B is not None:
            dxdtheta = dxdtheta.subs(self.__B, B)
        if theta is not None:
            dxdtheta = dxdtheta.subs(self.__theta, theta)
        return dxdtheta
    
    @property
    def a(self):
        return copy(self.__a)
    @property
    def b(self):
        return copy(self.__b)
    @property
    def B(self):
        return copy(self.__B)
    @property
    def theta(self):
        return copy(self.__theta)
    @property
    def x(self):
        return deepcopy(self.__x)
    @property
    def dxdtheta(self):
        return deepcopy(self.__dxdtheta)


class BackgroundStructure(object):
    __xisymb = abc.xi

    def __init__(self, gasproperties, dlogOmegadlogR=Rational(-3,2), theta=abc.theta):
        self.__gasproperties = gasproperties
        for k, v in self.__gasproperties.__dict__.items():
            self.__dict__[k] = deepcopy(v)
        self.__dlogOmegadlogR = dlogOmegadlogR
        self.__q = self.__dlogOmegadlogR**2
        self.__criticalPr = 1. / self.__q
        self.__theta = theta
        self.__B_expr = self.Pr / self.cV * self.__xisymb**2 * self.__q

        if self.Pr > self.__criticalPr:
            raise ValueError( 'Prandtl number Pr should be not greater than critical value dlogOmegadlogR**-2' )
        
        self.__symbolic = BackgroundStructureSymbolic()
        self.__x_expr_with_xi = self.__symbolic.x_subs(a=self.a, b=self.b, B=self.__B_expr)
        self.__dxdtheta_expr_with_xi = self.__symbolic.dxdtheta_subs(a=self.a, b=self.b, B=self.__B_expr)

        self.__thetabound = self.__find_thetabound()
        self.__xi = self.__find_xi()
        
        self.__x = self.__x_expr_with_xi.subs(self.__xisymb, self.__xi).evalf()
        self.__dxdtheta = self.__dxdtheta_expr_with_xi.subs(self.__xisymb, self.__xi).evalf()

        self.__xbound = float( self.__x.subs(self.__theta, self.__thetabound).evalf() )
        self.__B = float( self.__B_expr.subs(self.__xisymb, self.__xi).evalf() )

        self.__dlnlmbddtheta = - 1 / self.__theta - self.__xi**2 * self.gamma * self.__x / self.__theta * self.__dxdtheta

    @lru_cache(maxsize=1)
    def get_lmbd(self):
        return deepcopy( self.__get_lmbd() )

    @lru_cache(maxsize=1)
    def __get_lmbd(self):
        ab1 = self.a + self.b + 1

        first_term_lnlmbd = - ln(self.__theta)
        coef_for_23_terms_lnlmbd = self.__xi**2 * self.gamma * ab1 / (2 * self.__B) * _hyp_2F1_for_x(self.a, self.b, 1) / (self.a + 1)
        if self.a == 0:
            second_term_lnlmbd = 2 / (self.b + 1) *\
                ln( self.__theta**(self.b/2 + Rational(1,2)) / ( 1 + sqrt(1 - self.__theta**(self.b+1)) ) )
        else:
            second_term_lnlmbd = self.__theta**self.a / self.a *\
                hyper(
                    ( Rational(1, 2), self.a/ab1, ),
                    ( (2*self.a + self.b + 1)/ab1, ),
                    self.__theta**ab1
                )
        third_term_lnlmbd = - 1 / (self.a+1) / sqrt(pi) *\
            gamma( (2*self.a + 1) / ab1 ) * gamma( (3*self.a + self.b + 3)/(2*ab1) ) / gamma( (self.a + 1)/ab1 ) / gamma( (3*self.a + self.b + 2)/ab1 ) *\
            self.__theta**(2*self.a+1) *\
            hyper(
                ( 1, (3*self.a + self.b + 3)/(2*ab1), (2*self.a + 1)/ab1, ),
                ( (2*self.a + self.b + 2)/ab1, (3*self.a + self.b + 2)/ab1, ),
                self.__theta**ab1
            )

        lnlmbd = first_term_lnlmbd + coef_for_23_terms_lnlmbd * (second_term_lnlmbd + third_term_lnlmbd)
        lnlmbd = lnlmbd - lnlmbd.subs(self.__theta, 1.).evalf()

        lmbd = exp(lnlmbd)
        return lmbd

    @lru_cache(maxsize=1)
    def get_p(self):
        return deepcopy( self.__get_p() )

    @lru_cache(maxsize=1)
    def __get_p(self):
        p = self.__get_lmbd() * self.__theta
        return p

    def __find_xi(self):
        x_xi_expr = self.__x_expr_with_xi * self.__xisymb
        xi2_expr = 2 * self.cV * self.__theta + x_xi_expr**2
        xi2 = float( xi2_expr.subs(self.__theta, self.__thetabound).evalf() )
        return np.sqrt(xi2)

    def __find_thetabound(self):
        if self.a == 0 and self.b == 0:
            return 0.
        
        self.__expression_to_minimize = (
                self.__x_expr_with_xi * self.__dxdtheta_expr_with_xi * self.__xisymb**2 / self.cV + 1.
            ).cancel()
        self.__lambda_to_minimize = sympy.lambdify( self.__theta, self.__expression_to_minimize )
        thetabound = scipy.optimize.brentq(
            self.__func_to_minimize,
            0., 1.,
        )
        return thetabound

    def __func_to_minimize(self, y):
        if y == 1:
            return - np.sign( self.__func_to_minimize(0.) ) * np.inf
        return self.__lambda_to_minimize(y)

    @property
    def gasproperties(self):
        return deepcopy(self.__gasproperties)
    @property
    def dlogOmegadlogR(self):
        return copy(self.__dlogOmegadlogR)
    @property
    def theta(self):
        return copy(self.__theta)
    @property
    def xisymb(self):
        return copy(self.__xisymb)
    @property
    def x(self):
        return deepcopy(self.__x)
    @property
    def dxdtheta(self):
        return deepcopy(self.__dxdtheta)
    @property
    def lmbd(self):
        return self.get_lmbd()
    @property
    def dlnlmbddtheta(self):
        return deepcopy(self.__dlnlmbddtheta)
    @property
    def p(self):
        return self.get_p()
    @property
    def xbound(self):
        return self.__xbound
    @property
    def thetabound(self):
        return self.__thetabound
    @property
    def xi(self):
        return self.__xi
    @property
    def B(self):
        return self.__B
