"""Checks if a quantum state violates the PPT criterion."""

import numpy as np

from toqito.channels import partial_transpose
from toqito.matrix_props import is_positive_semidefinite


def is_ppt(mat: np.ndarray, sys: int = 2, dim: int | list[int] = None, tol: float = None) -> bool:
    r"""Determine whether or not a matrix has positive partial transpose :cite:`WikiPeresHorodecki`.

    Yields either :code:`True` or :code:`False`, indicating that :code:`mat` does or does not have
    positive partial transpose (within numerical error). The variable :code:`mat` is assumed to act
    on bipartite space.

    For shared systems of :math:`2 \otimes 2` or :math:`2 \otimes 3`, the PPT criterion serves as a
    method to determine whether a given state is entangled or separable. Therefore, for systems of
    this size, the return value :code:`True` would indicate that the state is separable and a value
    of :code:`False` would indicate the state is entangled.

    Examples
    ==========

    Consider the following matrix

    .. math::
        X =
        \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
        \end{pmatrix}.

    This matrix trivially satisfies the PPT criterion as can be seen using the
    :code:`|toqito⟩` package.

    .. jupyter-execute::

        from toqito.state_props import is_ppt
        import numpy as np
        mat = np.identity(9)
        is_ppt(mat)

    Consider the following Bell state:

    .. math::
        u = \frac{1}{\sqrt{2}}\left( |01 \rangle + |10 \rangle \right).

    For the density matrix :math:`\rho = u u^*`, as this is an entangled state
    of dimension :math:`2`, it will violate the PPT criterion, which can be seen
    using the :code:`|toqito⟩` package.

    .. jupyter-execute::

        from toqito.states import bell
        from toqito.state_props import is_ppt
        rho = bell(2) @ bell(2).conj().T
        is_ppt(rho)


    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param mat: A square matrix.
    :param sys: Scalar or vector indicating which subsystems the transpose
                should be applied on.
    :param dim: The dimension is a vector containing the dimensions of the
                subsystems on which :code:`mat` acts.
    :param tol: Tolerance with which to check whether `mat` is PPT.
    :return: Returns :code:`True` if :code:`mat` is PPT and :code:`False` if
             not.

    """
    if tol is None:
        tol = 1e-8  # Match is_separable's default

    if not isinstance(mat, np.ndarray) or mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input 'mat' must be a square 2D NumPy array.")

    mat_dim_total = mat.shape[0]
    processed_dim = None

    if dim is None:
        sqrt_len = np.sqrt(mat_dim_total)
        if np.isclose(sqrt_len, np.round(sqrt_len)):
            d_sub = int(np.round(sqrt_len))
            processed_dim = [d_sub, d_sub]
        else:
            raise ValueError(
                "If `dim` is None, matrix dimension must be a perfect square "
                "to infer symmetric bipartite dimensions. Please provide `dim`."
            )
    elif isinstance(dim, int):
        if mat_dim_total % dim != 0:
            raise ValueError(
                f"If `dim` is a scalar ({dim}), it must evenly divide matrix total dimension ({mat_dim_total})."
            )
        processed_dim = [dim, mat_dim_total // dim]
    elif isinstance(dim, (list, np.ndarray)):
        # Allow dim to be a single int in a list, e.g. [3] for a 9x9 matrix -> [3,3]
        if hasattr(dim, "__len__") and len(dim) == 1 and isinstance(dim[0], (int, np.integer)):
            d_A = int(dim[0])
            if (
                mat_dim_total % d_A != 0
            ):  # This check is only valid if we expect d_A * d_A = mat_dim_total or d_A * X = mat_dim_total
                # For [3] and 9x9, it implies [3,3]
                if d_A * d_A == mat_dim_total:
                    processed_dim = [d_A, d_A]
                elif mat_dim_total % d_A == 0:  # Infer second dim if first divides total
                    processed_dim = [d_A, mat_dim_total // d_A]
                else:
                    raise ValueError(
                        f"If `dim` is [{d_A}], it's ambiguous or does not properly factor matrix "
                        + "total dimension ({mat_dim_total})."
                    )
            else:  # d_A divides total_dim
                processed_dim = [d_A, mat_dim_total // d_A]

        elif hasattr(dim, "__len__") and len(dim) == 2:
            try:
                processed_dim = [int(d) for d in dim]
            except ValueError as e_val:
                raise ValueError(f"Elements of `dim` must be integers. Original error: {e_val}") from e_val

            if processed_dim[0] * processed_dim[1] != mat_dim_total:
                raise ValueError(
                    f"Product of dimensions in `dim` {processed_dim} ({processed_dim[0] * processed_dim[1]}) "
                    f"does not match matrix dimension {mat_dim_total}."
                )
        else:
            raise ValueError("`dim` must be None, an int, a list/array of one int, or a list/array of two ints.")
    else:
        raise TypeError(f"`dim` argument has an invalid type: {type(dim)}.")

    # Validate sys argument (expecting 1-based: 1 or 2 for bipartite)
    if sys not in {1, 2}:
        raise ValueError(
            f"For bipartite PPT check, 1-indexed `sys` must be 1 (first system) or 2 (second system). Got sys={sys}"
        )

    # Convert 1-based sys from user to 0-indexed list for partial_transpose
    pt_mat = partial_transpose(mat, sys=[sys - 1], dim=processed_dim)
    return is_positive_semidefinite(pt_mat, atol=tol, rtol=tol)
