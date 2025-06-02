"""Applies a quantum channel to an operator."""

import itertools

import numpy as np

from toqito.matrix_ops import vec
from toqito.perms import swap


def apply_channel(mat: np.ndarray, phi_op: np.ndarray | list[list[np.ndarray]]) -> np.ndarray:
    r"""Apply a quantum channel to an operator.

    (Section: Representations and Characterizations of Channels of :cite:`Watrous_2018_TQI`).

    Specifically, an application of the channel is defined as

    .. math::
        \Phi(X) = \text{Tr}_{\mathcal{X}} \left(J(\Phi)
        \left(\mathbb{I}_{\mathcal{Y}} \otimes X^{T}\right)\right),

    where

    .. math::
        J(\Phi): \text{T}(\mathcal{X}, \mathcal{Y}) \rightarrow
        \text{L}(\mathcal{Y} \otimes \mathcal{X})

    is the Choi representation of :math:`\Phi`.

    We assume the quantum channel given as :code:`phi_op` is provided as either the Choi matrix
    of the channel or a set of Kraus operators that define the quantum channel.

    This function is adapted from the QETLAB package.

    Examples
    ==========

    The swap operator is the Choi matrix of the transpose map. The following is a (non-ideal,
    but illustrative) way of computing the transpose of a matrix.

    Consider the following matrix

    .. math::
        X = \begin{pmatrix}
                1 & 4 & 7 \\
                2 & 5 & 8 \\
                3 & 6 & 9
            \end{pmatrix}

    Applying the swap operator given as

    .. math::
        \Phi =
        \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
         \end{pmatrix}

    to the matrix :math:`X`, we have the resulting matrix of

    .. math::
        \Phi(X) = \begin{pmatrix}
                        1 & 2 & 3 \\
                        4 & 5 & 6 \\
                        7 & 8 & 9
                   \end{pmatrix}

    Using :code:`|toqito⟩`, we can obtain the above matrices as follows.

    .. jupyter-execute::

     from toqito.channel_ops import apply_channel
     from toqito.perms import swap_operator
     import numpy as np
     test_input_mat = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
     apply_channel(test_input_mat, swap_operator(3))

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If matrix is not Choi matrix.
    :param mat: A matrix.
    :param phi_op: A superoperator. :code:`phi_op` should be provided either as a Choi matrix,
                   or as a list of numpy arrays with either 1 or 2 columns whose entries are its
                   Kraus operators.
    :return: The result of applying the superoperator :code:`phi_op` to the operator :code:`mat`.

    """
    # Both of the following methods of applying the superoperator are much faster than naively
    # looping through the Kraus operators or constructing eigenvectors of a Choi matrix.

    # The superoperator was given as a list of Kraus operators:
    if isinstance(phi_op, list):
        s_phi_op = [len(phi_op), len(phi_op[0])]

        phi_0_list = []
        phi_1_list = []

        # Map is completely positive if input is given as:
        # 1. [K1, K2, .. Kr]
        # 2. [[K1], [K2], .. [Kr]]
        # 3. [[K1, K2, .. Kr]] and r > 2
        if isinstance(phi_op[0], np.ndarray):
            phi_0_list = phi_op
        elif s_phi_op[1] == 1 or (s_phi_op[0] == 1 and s_phi_op[1] > 2):
            phi_0_list = list(itertools.chain(*phi_op))
        else:
            # Input is given as: [[A1, B1], [A2, B2], .. [Ar, Br]]
            phi_0_list = [k_mat[0] for k_mat in phi_op]
            phi_1_list = [k_mat[1].conj().T for k_mat in phi_op]

        if not phi_1_list:
            phi_1_list = [k_mat.conj().T for k_mat in phi_0_list]

        k_1 = np.concatenate(phi_0_list, axis=1)
        k_2 = np.concatenate(phi_1_list, axis=0)

        a_mat = np.kron(np.identity(len(phi_0_list)), mat)
        return k_1 @ a_mat @ k_2

    # The superoperator was given as a Choi matrix:
    if isinstance(phi_op, np.ndarray):
        # print(f"\n  DEBUG APPLY_CHANNEL (Choi Case):")
        # print(f"    Input mat shape: {mat.shape}")
        # print(f"    Input phi_op shape: {phi_op.shape}")

        mat_size = np.array(list(mat.shape))
        phi_size_factor_as_float_array = np.array(list(phi_op.shape), dtype=float) / mat_size

        # print(f"    mat_size: {mat_size}")
        # print(f"    phi_size_factor_as_float_array: {phi_size_factor_as_float_array}")

        if not np.allclose(phi_size_factor_as_float_array, np.round(phi_size_factor_as_float_array)):
            print("    ERROR: phi_size_factor not all integers!")  # Should not happen
            raise ValueError("Choi matrix dimensions are not integer multiples of the input matrix dimensions.")
        phi_size_factor_int_array = np.round(phi_size_factor_as_float_array).astype(int)
        # print(f"    phi_size_factor_int_array (interpreted d_out factors): {phi_size_factor_int_array}")

        # phi_size[0] is effectively d_out_rows_factor, phi_size[1] is d_out_cols_factor
        d_out_rows_factor = phi_size_factor_int_array[0]
        d_out_cols_factor = phi_size_factor_int_array[1]

        # This is where the interpretation of QETLAB's logic for a_mat is:
        # a_mat = kron(vec(X)^T, I_d_out_rows)
        a_mat = np.kron(vec(mat).T[0], np.identity(d_out_rows_factor))
        # print(f"    vec(mat).T[0] shape: {vec(mat).T[0].shape}")
        # print(f"    np.identity(d_out_rows_factor) shape: {np.identity(d_out_rows_factor).shape}")
        # print(f"    a_mat shape: {a_mat.shape}")

        swap_dims = np.array(
            [
                [mat_size[1], d_out_cols_factor],  # [cols_X, d_out_cols]
                [mat_size[0], d_out_rows_factor],
            ],  # [rows_X, d_out_rows]
            dtype=int,
        )
        # print(f"    swap_dims for permute_systems: {swap_dims.tolist()}")

        b_mat_intermediate = swap(
            phi_op.T,  # J(OverallMap)^T
            [1, 2],  # Swaps the two conceptual systems of J
            swap_dims,  # Tells how J is structured from 4 virtual subsystems
            True,  # multipartite=True means swap_dims defines structure for 2 systems, each made of 2 factors
        ).T  # Another transpose
        # print(f"    b_mat_intermediate (after swap and T) shape: {b_mat_intermediate.shape}")

        # Reshape dimensions for b_mat: (d_out_rows * rows_X * cols_X, d_out_cols)
        b_mat_reshape_dims = (d_out_rows_factor * mat_size[0] * mat_size[1], d_out_cols_factor)
        # print(f"    b_mat_reshape_dims: {b_mat_reshape_dims}")

        b_mat = np.reshape(b_mat_intermediate, b_mat_reshape_dims, order="F")
        # print(f"    b_mat shape: {b_mat.shape}")

        result = a_mat @ b_mat
        # print(f"    Final result shape from apply_channel: {result.shape}")
        return result
    raise ValueError("Invalid: The variable `phi_op` must either be a list of Kraus operators or as a Choi matrix.")
