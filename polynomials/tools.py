#!/usr/bin/env python3


import numpy as np
import numpy.matlib as ml
import sympy, sympy.abc


def plotPolynomial(cs, x=None, filename='polynomial.pdf'):
    """
    Plot polynomial sum( cs[i] * x**i ) and save it to PDF file

    Parameters
    ----------
    cs : array-like
        1-D array with polynomial coefficients
    x : array-like, optional
        Where to plot (1-D array). If None than use np.linspace(0, 1, 1000).
        Default is None
    filename : str, optional
        Filename of output PDF plot. Default is 'polynomial.pdf'

    """
    if x is None:
        x = np.linspace(0, 1, 1000)
    else:
        x = np.asarray(x)
    
    cs = np.asarray(cs)
    y = np.zeros_like(x, dtype=cs.dtype)

    for i in range(cs.shape[0]):
        y += cs[i] * x**i

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.cla()
    if (cs.imag != cs).any():
        plt.plot( x, y.real, 'r-', x, y.imag, 'b-' )
        plt.legend(( 'Re', 'Im' ))
    else:
        plt.plot( x, y, 'r-' )
        plt.legend('Re')
    plt.xlabel('x')
    plt.ylabel('Polynomial')
    plt.savefig( filename, format='pdf' )



def roots_of_matrix(M, x=sympy.abc.x):
    '''
    Find roots of det( M(x) ) = 0 equation, where M(x) = A + B*x.
    
    Parameters
    ----------
    M : sympy.Matrix
        Square matrix depends linearly on only one variable
    x : sympy.Symbol, optional
        Symbol which m depends on

    Returns
    -------
    numpy.ndarray
        1-D array of roots
    '''
    M_wo_x = M.replace(x, 0.)
    A = ml.empty(M.shape, dtype=np.complex)
    for (i,j), element in np.ndenumerate(M_wo_x):
        A[i,j] = complex(element)
    
    M_coeffs_x = (M - M_wo_x).replace(x, 1.)
    B = ml.empty(M.shape, dtype=np.complex)
    for (i,j), element in np.ndenumerate(M_coeffs_x):
        B[i,j] = complex(element)

    C = - A.I * B

    evs = np.linalg.eigvals(C)
    roots = 1 / evs[ np.nonzero(evs) ]
    return roots
