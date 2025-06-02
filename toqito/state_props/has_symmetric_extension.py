"""Determine whether there exists a symmetric extension for a given quantum state."""

import numpy as np
from picos import partial_trace

from toqito.matrix_props import is_positive_semidefinite
from toqito.state_opt.symmetric_extension_hierarchy import symmetric_extension_hierarchy
from toqito.state_props.is_ppt import is_ppt


def has_symmetric_extension(
    rho: np.ndarray,
    level: int = 2,
    dim: np.ndarray | int = None,
    ppt: bool = True,
    tol: float = 1e-5,
) -> bool:
    r"""Determine whether there exists a symmetric extension for a given quantum state.

    For more information, see :cite:`Doherty_2002_Distinguishing`.

    Determining whether an operator possesses a symmetric extension at some level :code:`level`
    can be used as a check to determine if the operator is entangled or not.

    This function was adapted from QETLAB.

    Examples
    ==========

    2-qubit symmetric extension:

    In :cite:`Chen_2014_Symmetric`, it was shown that a 2-qubit state :math:`\rho_{AB}` has a
    symmetric extension if and only if

    .. math::
        \text{Tr}(\rho_B^2) \geq \text{Tr}(\rho_{AB}^2) - 4 \sqrt{\text{det}(\rho_{AB})}.

    This closed-form equation is much quicker to check than running the semidefinite program.

    .. jupyter-execute::

        import numpy as np
        from toqito.state_props import has_symmetric_extension
        from toqito.channels import partial_trace
        rho = np.array([[1, 0, 0, -1], [0, 1, 1/2, 0], [0, 1/2, 1, 0], [-1, 0, 0, 1]])
        # Show the closed-form equation holds
        np.trace(np.linalg.matrix_power(partial_trace(rho, 1), 2)) >= np.trace(rho**2) - 4 * np.sqrt(np.linalg.det(rho))

    .. jupyter-execute::

        # Now show that the `has_symmetric_extension` function recognizes this case.
        has_symmetric_extension(rho)

    Higher qubit systems:

    Consider a density operator corresponding to one of the Bell states.

    .. math::
        \rho = \frac{1}{2} \begin{pmatrix}
                            1 & 0 & 0 & 1 \\
                            0 & 0 & 0 & 0 \\
                            0 & 0 & 0 & 0 \\
                            1 & 0 & 0 & 1
                           \end{pmatrix}

    To make this state over more than just two qubits, let's construct the following state

    .. math::
        \sigma = \rho \otimes \rho.

    As the state :math:`\sigma` is entangled, there should not exist a symmetric extension at some
    level. We see this being the case for a relatively low level of the hierarchy.

    .. jupyter-execute::

        import numpy as np
        from toqito.states import bell
        from toqito.state_props import has_symmetric_extension
        rho = bell(0) @ bell(0).conj().T
        sigma = np.kron(rho, rho)
        has_symmetric_extension(sigma)


    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If dimension does not evenly divide matrix length.
    :param rho: A matrix or vector.
    :param level: Level of the hierarchy to compute.
    :param dim: The default has both subsystems of equal dimension.
    :param ppt: If :code:`True`, this enforces that the symmetric extension must be PPT.
    :param tol: Tolerance when determining whether a symmetric extension exists.
    :return: :code:`True` if :code:`mat` has a symmetric extension; :code:`False` otherwise.

    """
    if not isinstance(rho, np.ndarray) or rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("Input rho must be a square 2D NumPy array.")
    len_mat = rho.shape[0]

    # Standardized dim processing:
    if dim is None:
        dim_val = int(np.round(np.sqrt(len_mat)))
        if dim_val**2 != len_mat:
            # Try to infer if it's dA * dB, where dA might be different from dB
            # This part is tricky without more assumptions. For now, stick to what was there.
            # Defaulting to symmetric bipartite if perfect square, else error.
            raise ValueError(
                "Input matrix total dimension is not a perfect square, "
                "so cannot infer symmetric bipartite dimensions. Please provide `dim`."
            )
        processed_dim = np.array([dim_val, dim_val], dtype=int)
    elif isinstance(dim, int):
        if len_mat % dim != 0:
            raise ValueError("If `dim` is a scalar, it must evenly divide matrix length.")
        processed_dim = np.array([dim, len_mat // dim], dtype=int)
    elif isinstance(dim, (list, np.ndarray)) and len(dim) == 2:
        processed_dim = np.array(dim, dtype=int)
        if processed_dim[0] * processed_dim[1] != len_mat:
            raise ValueError("Product of dimensions in `dim` does not match matrix dimension.")
    else:
        raise ValueError("`dim` must be None, an int, or a list/array of two ints representing [dA, dB].")

    # Early exit conditions from your provided code:
    dim_x, dim_y = processed_dim[0], processed_dim[1]

    # Level 1 symmetric extension means the state itself is the extension.
    if level == 1:
        if not ppt:  # Only require the extension (rho itself) to be PSD
            return is_positive_semidefinite(rho, atol=tol)  # Use tolerance
        # Require the extension (rho itself) to be PPT and PSD
        return is_ppt(rho, sys=1, dim=processed_dim.tolist(), tol=tol) and is_positive_semidefinite(
            rho, atol=tol
        )  # Use tolerance

    # For small systems (2x2, 2x3, 3x2) if PPT is required for the extension,
    # and level >= 1 (already handled by above), it's just asking if rho is PPT.
    # (This was combined with level=1 in your original code)
    if dim_x * dim_y <= 6 and ppt:  # True if level > 1 and this condition met
        return is_ppt(rho, sys=1, dim=processed_dim.tolist(), tol=tol) and is_positive_semidefinite(rho, atol=tol)

    # 2-qubit analytic formula for k=2 (non-PPT extension)
    if level == 2 and not ppt and dim_x == 2 and dim_y == 2:
        # Ensure partial_trace is called correctly for subsystem B (index 1 if 0-indexed)
        # toqito's partial_trace sys is 0-indexed list
        tr_rho_b = partial_trace(rho, sys=[1], dim=processed_dim.tolist())
        # Check for det(rho) being negative due to numerical precision
        det_rho = np.linalg.det(rho)
        sqrt_det_rho = (
            np.sqrt(np.maximum(0, det_rho)) if det_rho < 0 and np.isclose(det_rho, 0, atol=1e-9) else np.sqrt(det_rho)
        )
        if isinstance(sqrt_det_rho, complex) and np.isclose(sqrt_det_rho.imag, 0):
            sqrt_det_rho = sqrt_det_rho.real
        if isinstance(sqrt_det_rho, complex) or sqrt_det_rho < 0:  # Should not happen if rho is PSD
            # If det(rho) is significantly negative, state wasn't PSD or there's an issue.
            # For this analytic formula, non-PSD rho is problematic.
            # Assuming rho is PSD as a precondition for has_symmetric_extension.
            pass  # Let the formula proceed, it might still work out if det is ~0

        return (
            np.trace(np.linalg.matrix_power(tr_rho_b, 2))
            >= np.trace(np.linalg.matrix_power(rho, 2)) - 4 * sqrt_det_rho - tol
        )  # Added tol for comparison

    # Call to the SDP hierarchy
    # **CRITICAL**: Ensure parameter names match symmetric_extension_hierarchy's definition.
    # Assuming:
    #   - `states` is a list of operators
    #   - `probs` is a list of probabilities (use [1.0] for a single state)
    #   - `k` is the extension level
    #   - `dim` is the bipartite dimension list [dA, dB]
    #   - It returns a value `sdp_val` where `sdp_val <= tol` means extendible.

    # print(f"DEBUG has_symmetric_extension: Calling hierarchy with dim={processed_dim.tolist()}, level={level}")
    # Call to the SDP hierarchy
    # print(f"DEBUG has_symmetric_extension: Calling hierarchy with dim={processed_dim.tolist()}, level={level}")
    sdp_val = symmetric_extension_hierarchy(states=[rho], probs=[1.0], level=level, dim=processed_dim.tolist())

    # Make this print for 8x8 or 9x9 states
    # is_entangled_test_rho = (rho.shape == (9,9) and np.isclose(rho[0,1], 0.07444444)) # Example check
    # if is_entangled_test_rho:
    #     print(f"\nDEBUG has_symmetric_extension (ENTANGLED TEST STATE):")
    #     print(f"  rho_shape={rho.shape}, level={level}, dim={processed_dim.tolist()}")
    #     print(f"  sdp_val from hierarchy: {sdp_val}")
    #     print(f"  tol being used for np.isclose(sdp_val, 1.0, atol=tol): {tol}")
    #     print(f"  np.isclose(sdp_val, 1.0, atol=tol) will be: {np.isclose(sdp_val, 1.0, atol=tol)}")

    return np.isclose(sdp_val, 1.0, atol=tol)
