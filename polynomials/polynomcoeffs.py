#!/usr/bin/env python3


import copy
import logging
import numpy as np
import sympy
from scipy import optimize
from sympy.abc import xi


def _xreplace_in_expr(expr, rule):
    return expr.xreplace(rule)

_ufunc_xreplace = np.frompyfunc(
    _xreplace_in_expr,
    2,
    1
)


class DifEq(object):
    """
    Solver of general Sturm–Liouville problem for homogeneous different
    equation system of one variable xi.
    Solver represents different equations and boundary conditions as polynomial
    serials of order k_max and equates coefficients with similar orders of the
    variable.

    Parameters
    ----------
    fs : numpy.ndarray, dtype=sympy.Function
        Array of unknown functions of sympy.abc.xi
    eqs : array-like 
        Array of tuples. Each tuple is a pair of a sympy expression for one of
        the equations and an order of this equation
    boundconds : array-like 
        Array of tuples. Each tuple is a pair of a sympy expression for one of
        the boundary conditions and a coordinate of variable for this bound 
    ev_symbol : sympy.Symbol
        Symbol of eigenvalue, it should enter self.__eqs and/or self.boundcond
    k_max : int
        Order of polynomial representations of functions fs
    replace_rule : dict-like, optional
        Dictionary with values of eqs and boundconds parameters (except
        ev_symbol) that you would like to replace with numerical values before
        any calculations. Key is sympy.Symbol and value is float of complex
        number. These variables cannot be modified. Default is an empty
        frozenset
    parameters : dict-like, optional
        Most like replace_rule but could be modified. replace_rule and
        parameters must contains all the symbols of eqs and boundconds except
        eqs and ev_symbol. Default is an empty frozenset
    dumpfile : str, optional
        If not None than try to load self.matrix from dumpfile. All parameters
        except replace_rule should be the same as was in DifEq object that used
        to create dump via self.dump(dumpfile). Default is None

    Attributes
    ----------
    fs_dict : dict
        'Inversed' self.fs array: keys are items of self.fs and values are its
        indexes
    k_max : int
        Order of polynomial representations of function self.__fs
    matrix : sympy.Matrix
        Square matrix of order (k_max+1)*fs.len found in self.computeMatrix
    ev : float or complex
        Eigenvalue found in self.getEigenValue
    A : numpy.ndarray
        Square matrix that represents self.matrix but with numerical values of
        found eigenvalue and parameters. Computed in self.getEigenValue
    cs_dict : dict
        Coefficients of polynomial representation of fs found in
        self.computeEigenFunctions

    Methods
    -------
    findEigenValue(ev0, force_complex=False)
        Compute eigenvalue of the problem
    computeEigenFunctions(f_symbol=None, calibr_k=None)
        Compute eigenfunctions of the problem.
        Should be called after self.findEigenValue
    setParams(parameters)
        Reset self.parameters
    dump(dumpfile)
        Dump self.matrix to dumpfile

    Notes
    -----
    Object of the class cannot be used competitively!

    Examples
    --------
    Solution of harmonic oscillator equation but with wrong sign:
    f''(x) - omega^2 * f(x) = 0,
    f'(0) = 0, f(1) = 0.

    >>> import sympy, sympy.abc
    >>> import numpy as np
    >>> from polynomials import DifEq
    >>> omega = sympy.abc.omega
    >>> k_max = 6  # Series asymptotic is O(xi**(k_max+1))
    >>> # Define unknown function name and arrays with unknown functions
    >>> f = sympy.Function('f')
    >>> fs = np.array( [f], dtype=sympy.Function )
    >>> # Structure with equation and its order
    >>> eqs = [
    ... (
    ...     -omega**2 * f(xi) + sympy.diff(f(xi),xi,2),
    ...     2
    ... )
    ... ]
    >>> # Structure with boundaries
    >>> boundconds = [
    ... (
    ...     sympy.diff(f(xi),xi,1),
    ...     0.
    ... ),
    ... (
    ...     f(xi),
    ...     1.
    ... )
    ... ]
    >>> # Create object
    >>> de = DifEq(fs, eqs, boundconds, omega, k_max)
    >>> # Get value omega closest to imaginary union
    >>> omega_num = de.findEigenValue( 1.j )
    >>> # Compare with (pi/2 * 1i)
    >>> np.testing.assert_almost_equal(omega_num, 0.5j * np.pi, decimal=2)
    >>> # Get polynomial coefficients
    >>> cs_dict = de.computeEigenFunctions( f_symbol=f, calibr_k=0 )
    >>> # Compare the result with Taylor series for cos( pi/2 * x )
    >>> pi_2 = np.pi / 2
    >>> fact = np.math.factorial
    >>> cos_series = list(
    ...     (i+1)%2 * pi_2**i * (-1)**(i//2) / fact(i) for i in range(k_max+1)
    ... )
    >>> np.testing.assert_array_almost_equal(
    ...     cos_series,
    ...     cs_dict[f],
    ...     decimal=2
    ... )  
    
    """

    __M = None
    __parameters = None
    __ev = None
    __A = None
    __detA = None
    __csval_dict = None
    __Mfunc = None
    __detMfunc = None

    def __init__(self,
                 fs,
                 eqs,
                 boundconds,
                 ev_symbol,
                 k_max,
                 replace_rule=frozenset(),
                 parameters=frozenset(),
                 dumpfile=None,):
        self.__fs = copy.deepcopy( fs )
        self.__fs_dict = dict( (fs[f_idx], f_idx) for f_idx in range(fs.shape[0]) )
        self.__eqs = np.array(
            eqs,
            dtype=[ ('eq', 'object'), ('order', 'int') ]
        )
        self.__eqs['eq'] = _ufunc_xreplace( self.__eqs['eq'], replace_rule )
        self.__boundconds = np.array(
            boundconds,
            dtype=[ ('eq', 'object'), ('xi', 'float') ]
        )
        self.__boundconds['eq'] = _ufunc_xreplace( self.__boundconds['eq'], replace_rule )
        self.__ev_symbol = ev_symbol
        self.__k_max = k_max

        if dumpfile is not None:
            try:
                self.__load(dumpfile, parameters)
                return
            except Exception as e:
                logging.warning('Cannot load from dumpfile:\n{}\nNormal creation of the object'.format(e))
        self.__computeMatrix()
        self.setParams(parameters)

    def __computeMatrix(self):
        """
        Compute square sympy.Matrix of order (k_max+1)*fs.len with coefficients
        before polynomial coefficients c_k^f in matrix representation of
        differential equation system. The lines represents coefficient before
        xi**n in one of equations or boundary conditions. The rows represents
        c_k^f.

        """
        self.__cs = np.array( [
            [
                sympy.symbols( 'c_{f}_{k}'.format(
                    f = str(f),
                    k = k
                ) )
            for k in range(self.__k_max+1)]
        for f in self.__fs
        ] )

        self.__ps = ( self.__cs * xi**np.arange(self.__k_max+1) ).sum(axis=1)
        self.__func2poly_dict = dict(
            (self.__fs[i_f](xi), self.__ps[i_f]) for i_f in range(self.__fs.shape[0])
        )

        self.__peqs = []
        for eq in self.__eqs:
            # Replace functions with corresponding polynomials
            peq = eq['eq'].xreplace(self.__func2poly_dict).doit()
            # Expand into series
            max_order = self.__k_max + 1 - eq['order']
            peq = peq.series(x=xi, n=max_order).removeO() 

            self.__peqs.append( peq.expand() )
        for bc in self.__boundconds:
            # Replace functions with corresponding polynomials
            peq = bc['eq'].xreplace(self.__func2poly_dict).doit()
            # Expand into series
            peq = peq.series(x=xi, n=self.__k_max+1).removeO()
            # Evaluate expression at boundary
            peq = peq.replace(xi, bc['xi'])

            self.__peqs.append( peq.expand() )
        self.__peqs = np.array(self.__peqs)

        self.__lineqs = []
        for i_peq in range(self.__peqs.shape[0]):
            peq = copy.copy( self.__peqs[i_peq] )

            order = 0
            component = peq.replace(xi, 0)
            peq -= component
            self.__lineqs.append(component)
            while peq != 0:
                order += 1
                component = peq.coeff(xi**order)
                peq = (peq - component * xi**order).expand()
                self.__lineqs.append(component)            
            
        self.__M = []
        for lineq in self.__lineqs:
            Aline = np.empty_like( self.__cs.flat, dtype=object )
            for i_c in range(self.__cs.size):
                c = self.__cs.flat[i_c]
                Aline[i_c] = lineq.coeff(c)
            self.__M.append(Aline)
        self.__M = sympy.Matrix(self.__M)

    def __computeInitParametersHash(self):
        """
        Compute hash from all self.__init__ parameters (except replace_rule
        and dumpfile) after applying replace_rule to eqs and boun_conds.
        Resulting hashsum is used to identify dump file.

        Returns
        -------
        bytes
            Hex digest of md5 hash

        """
        import hashlib
        
        attrs_for_hash = (
            'fs',
            'eqs',
            'boundconds',
            'ev_symbol',
            'k_max',
        )
        self.__init_params_hash = hashlib.md5(
            bytes(
                tuple(
                    self.__getattribute__(attr) for attr in attrs_for_hash
                ).__repr__(),
                encoding='utf-8'
            )
        ).hexdigest()
        return self.__init_params_hash

    def dump(self, dumpfile):
        """
        Dump self.matrix to dumpfile

        Parameters
        ----------
        dumpfile : str
            Filename where to dump
        
        """
        import pickle
        
        iph = self.__computeInitParametersHash()
        dict2dump = {
            'init_params_hash': iph,
            'M': self.__M
        }
        with open(dumpfile, 'wb') as fh:
            pickle.dump( dict2dump, fh )

    def __load(self, dumpfile, parameters):
        """
        Load self.__M from dump file. If init parameters in the object from the
        dump differs from current object parameters than raise AttributionError

        Parameters
        ----------
        dumpfile : str
            Name of the dumpfile
        parameters : dict-like
            Value to set self.__parameters

        """
        import pickle
        
        iph = self.__computeInitParametersHash()
        with open(dumpfile, 'rb') as fh:
            loaded_dict = pickle.load(fh)
            if loaded_dict['init_params_hash'] != iph:
                raise AttributeError('Cannot load object with different input parameters')
            self.__M = loaded_dict['M']
            self.setParams(parameters)

    def setParams(self, parameters):
        """
        Reset self.parameters

        Parameters
        ----------
        parameters : dict-like
            Dictionary sympy.Symbol - numerical value for all parameters of the
            problem

        Returns
        -------
        None

        """
        if parameters == self.__parameters  and  not (self.__Mfunc is None):
            return

        self.__parameters = parameters
        
        if len(parameters) == 0:
            self.__M_with_eigen_only = self.__M
        else:
            self.__M_with_eigen_only = copy.deepcopy( self.__M )
            self.__M_with_eigen_only = self.__M_with_eigen_only.xreplace(parameters)

        self.__Mfunc = sympy.lambdify(
            self.__ev_symbol,
            self.__M_with_eigen_only,
            modules=[{'ImmutableMatrix': np.array}, 'numpy']
        )

        # Should we use np.linalg.slogdet instead of det?
        def detMfunc(x):
            A_tmp = self.__Mfunc(x)
            A_tmp /= np.abs(A_tmp.diagonal()).mean()
            return np.linalg.det(A_tmp)
        self.__detMfunc = detMfunc

    def findEigenValue(self,
                       ev0,
                       force_complex=False,):
        """
        Compute eigenvalue of the problem.
        Replaces parameters of the system with numerical values and find 
        eigenvalue that solves det(M) = 0.

        Parameters
        ----------
        ev0 : float or complex
            Initial guess
        force_complex : bool
            Force search eigenvalue as complex number, default is False.
            If M doesn't contain sympy.I, ev0 is real and force_complex is
            False than try to find eigenvalue as real number.

        Returns
        -------
        float or complex
            Eigenvalue of the problem

        """
        self.__csval_dict = None

        # Check should we search ev as complex or as real
        if force_complex  or  (hasattr(ev0, 'imag') and ev0.imag != 0)  or  self.__M.replace(sympy.I, 0) != self.__M:
            minimize_result = optimize.minimize(
                lambda x: abs( self.__detMfunc( x[0] + 1j*x[1] ) ),
                np.array([ev0.real, ev0.imag]),
                method='Nelder-Mead',
            )
            self.__ev = minimize_result.x[0] + 1j * minimize_result.x[1]
            self.__detA = minimize_result.fun
        else:
            self.__ev = optimize.newton(
                self.__detMfunc,
                ev0
            )
            self.__detA = self.__detMfunc(self.__ev)
        
        self.__A = self.__Mfunc(self.__ev)
        
        return self.__ev

    def getDeterminantMap(self, grid):
        """
        Calculate absolute values of self.matrix determinate on given grid
        of eigenvalue candidates

        Parameters
        ----------
        grid : numpy.ndarray
            Array with eigenvalue candidates

        Returns
        -------
        numpy.ndarray
            Array with the same shape as grid with calculated absolute values
            of self.matrix determinate

        """
        self.__ufunc_absdetM = np.vectorize(
            lambda x: abs( self.__detMfunc(x) ),
            otypes='g'
        )

        return self.__ufunc_absdetM( grid )

    def drawDeterminantMap(self, re, im, filename='det_map.pdf'):
        """
        Draw absolute values of self.matrix determinate on given grid of real
        and imaginary parts of possible eigenvalues as a contour map. Requires
        matplotlib

        Parameters
        ----------
        re : numpy.ndarray
            1-D array with the values along real axis
        im : numpy.ndarray
            1-D array with the values along imaginary axis
        filename : str, optional
            Filename for output PDF plot

        Returns
        -------
        None

        """
        n_re = re.size
        n_im = im.size

        grid = np.dot(
            np.ones(( n_re, 1 )),
            re.reshape(1, n_re)
        ) +\
        1j * np.dot(
            im.reshape(n_im, 1),
            np.ones(( 1, n_im ))
        )

        dets = self.getDeterminantMap( grid )
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.cla()
        CS = plt.contour( re, im, np.log10(dets), max(n_re, n_im, 100)//5 )
        plt.clabel(CS, inline=1, fontsize=10)
        plt.xlabel(r'Re$(\omega / \Omega)$')
        plt.ylabel(r'Im$(\omega / \Omega)$')
        plt.title(r'$\log_{10}{\det(A)}$')
        plt.savefig( filename, format='pdf' )

    def computeEigenFunctions(self, f_symbol=None, calibr_k=None):
        """
        Compute eigenfunctions of the problem.
        Find coefficients of polynomial representation of eigenfunctions for the
        case when one of these coefficients equals unity.

        Parameters
        ----------
        f_symbol : sympy.Symbol
            Function for which we normalize all other functions.
            If not specified than use self.__fs[0]
        calibr_k : int
            Coefficient of xi**calibr_k is expected to be unity.
            If calibr_k isn't specified than expect free
            coefficient to be unity

        Returns
        -------
        dict
            Coefficients of polynomials series as a dictionary where keys
            are elements of self.__fs and values are numpy.ndarrays of
            corresponding coefficients

        """
        if self.__ev is None:
            raise ValueError('Call self.getEigenValue first')

        if calibr_k is None:
            self.__calibr_c_idx = 0
        else:
            if f_symbol is None:
                f_symbol = self.__fs[0]
            self.__calibr_c_idx = self.__fs_dict[f_symbol] * (self.__k_max+1) + calibr_k
        
        self.__b = - self.__A[
            np.array( [i!=self.__calibr_c_idx for i in range(self.__A.shape[0])] ),
            self.__calibr_c_idx
        ]
        self.__Atrunc = self.__A[
            np.array([ i!=self.__calibr_c_idx and j!= self.__calibr_c_idx for i in range(self.__A.shape[0]) for j in range(self.__A.shape[1]) ]).reshape(self.__A.shape)
        ].reshape( (self.__A.shape[0]-1, self.__A.shape[1]-1) )
        self.__cs_num = np.linalg.solve(self.__Atrunc, self.__b)
        self.__cs_num = np.insert(self.__cs_num, self.__calibr_c_idx, 1.).reshape( (self.__fs.shape[0], self.__k_max+1) )
        
        self.__csval_dict = dict(
            (f_symbol, self.__cs_num[f_idx]) for f_symbol, f_idx in self.__fs_dict.items()
        )

        return self.__csval_dict.copy()

    @property
    def fs(self):
        return copy.deepcopy( self.__fs )

    @property
    def fs_dict(self):
        return self.__fs_dict.copy()

    @property
    def eqs(self):
        return copy.deepcopy( self.__eqs )

    @property
    def boundconds(self):
        return copy.deepcopy( self.__boundconds )

    @property
    def ev_symbol(self):
        return self.__ev_symbol

    @property
    def k_max(self):
        return self.__k_max

    @property
    def parameters(self):
        return self.__parameters.copy()

    @parameters.setter
    def parameters(self, new_parameters):
        self.setParams(new_parameters)

    @property
    def matrix(self):
        if self.__M is None:
            raise AttributeError("matrix hasn't been specified yet")
        return copy.deepcopy( self.__M )

    @property
    def ev(self):
        if self.__ev is None:
            raise AttributeError("ev hasn't been specified yet")
        return self.__ev

    @ev.setter
    def ev(self, value):
        self.__ev = value
        self.__csval_dict = None
        self.__A = self.__Mfunc(self.__ev)
        self.__detA = self.__detMfunc(self.__ev)

    @property
    def A(self):
        if self.__A is None:
            raise AttributeError("A hasn't been specified yet")
        return self.__A.copy()

    @property
    def detA(self):
        if self.__detA is None:
            raise AttributeError("detA hasn't been specified yet")
        return self.__detA

    @property
    def cs_dict(self):
        if self.__csval_dict is None:
            raise AttributeError("cs_dict hasn't been specified yet")
        return self.__csval_dict.copy()
