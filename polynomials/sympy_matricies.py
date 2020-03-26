# Patched version of sympy.matrices.matrices.det_bareis


import sympy


def det_bareis(self):
    """Compute matrix determinant using Bareis' fraction-free
    algorithm which is an extension of the well known Gaussian
    elimination method. This approach is best suited for dense
    symbolic matrices and will result in a determinant with
    minimal number of fractions. It means that less term
    rewriting is needed on resulting formulae.

    TODO: Implement algorithm for sparse matrices (SFF),
    http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.

    See Also
    ========

    det
    berkowitz_det
    """
    if not self.is_square:
        raise NonSquareMatrixError()
    if not self:
        return S.One

    M, n = self.copy().as_mutable(), self.rows

    if n == 1:
        det = M[0, 0]
    elif n == 2:
        det = M[0, 0]*M[1, 1] - M[0, 1]*M[1, 0]
    elif n == 3:
        det = (M[0, 0]*M[1, 1]*M[2, 2] + M[0, 1]*M[1, 2]*M[2, 0] + M[0, 2]*M[1, 0]*M[2, 1]) - \
              (M[0, 2]*M[1, 1]*M[2, 0] + M[0, 0]*M[1, 2]*M[2, 1] + M[0, 1]*M[1, 0]*M[2, 2])
    else:
        sign = 1  # track current sign in case of column swap

        for k in range(n - 1):
            # look for a pivot in the current column
            # and assume det == 0 if none is found
            if M[k, k] == 0:
                for i in range(k + 1, n):
                    if M[i, k]:
                        M.row_swap(i, k)
                        sign *= -1
                        break
                else:
                    return S.Zero

            # proceed with Bareis' fraction-free (FF)
            # form of Gaussian elimination algorithm
            for i in range(k + 1, n):
                for j in range(k + 1, n):
                    D = M[k, k]*M[i, j] - M[i, k]*M[k, j]

                    if k > 0:
                        D /= M[k - 1, k - 1]

                    # if D.is_Atom:
                    #     M[i, j] = D
                    # else:
                    #     M[i, j] = cancel(D)
                    M[i,j] = D

        det = sign*M[n - 1, n - 1]

    return det.expand()
