"""Applies a channel to a subsystem of an operator."""

import itertools

import numpy as np

from toqito.channel_ops import apply_channel
from toqito.perms import permute_systems
from toqito.states import max_entangled


def partial_channel(
    rho: np.ndarray,
    phi_map: np.ndarray | list[list[np.ndarray]],
    sys: int = 2,
    dim: list[int] | np.ndarray = None,
) -> np.ndarray:
    r"""Apply channel to a subsystem of an operator :cite:`Watrous_2018_TQI`.

    Applies the operator

    .. math::
        \left(\mathbb{I} \otimes \Phi \right) \left(\rho \right).

    In other words, it is the result of applying the channel :math:`\Phi` to the second subsystem
    of :math:`\rho`, which is assumed to act on two subsystems of equal dimension.

    The input :code:`phi_map` should be provided as a Choi matrix.

    This function is adapted from the QETLAB package.

    Examples
    ==========

    The following applies the completely depolarizing channel to the second
    subsystem of a random density matrix.

    .. jupyter-execute::

     import numpy as np
     from toqito.channel_ops import partial_channel
     from toqito.channels import depolarizing
     rho = np.array([
        [0.3101, -0.0220 - 0.0219j, -0.0671 - 0.0030j, -0.0170 - 0.0694j],
        [-0.0220 + 0.0219j, 0.1008, -0.0775 + 0.0492j, -0.0613 + 0.0529j],
        [-0.0671 + 0.0030j, -0.0775 - 0.0492j, 0.1361, 0.0602 + 0.0062j],
        [-0.0170 + 0.0694j, -0.0613 - 0.0529j, 0.0602 - 0.0062j, 0.4530]
     ])

     res = partial_channel(rho, depolarizing(2))

     np.set_printoptions(linewidth=150, suppress=False)
     print(res)



    The following applies the completely depolarizing channel to the first
    subsystem.

    .. jupyter-execute::

     import numpy as np
     from toqito.channel_ops import partial_channel
     from toqito.channels import depolarizing

     rho = np.array([
        [0.3101, -0.0220 - 0.0219j, -0.0671 - 0.0030j, -0.0170 - 0.0694j],
        [-0.0220 + 0.0219j, 0.1008, -0.0775 + 0.0492j, -0.0613 + 0.0529j],
        [-0.0671 + 0.0030j, -0.0775 - 0.0492j, 0.1361, 0.0602 + 0.0062j],
        [-0.0170 + 0.0694j, -0.0613 - 0.0529j, 0.0602 - 0.0062j, 0.4530]
     ])

     res = partial_channel(rho, depolarizing(2))
     np.set_printoptions(linewidth=150, suppress=False)
     print(res)


    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If Phi map is not provided as a Choi matrix or Kraus
                        operators.
    :param rho: A matrix.
    :param phi_map: The map to partially apply.
    :param sys: Scalar or vector specifying the size of the subsystems.
    :param dim: Dimension of the subsystems. If :code:`None`, all dimensions
                are assumed to be equal.
    :return: The partial map :code:`phi_map` applied to matrix :code:`rho`.

    """
    if dim is None:
        dim = np.round(np.sqrt(list(rho.shape))).conj().T * np.ones(2)
    if isinstance(dim, list):
        dim = np.array(dim)

    # Force dim to be a row vector.
    if dim.ndim == 1:
        dim = dim.T.flatten()
        dim = np.array([dim, dim])

    # We expect rho.shape to be (8,8) for the problematic 2x4 case
    # if rho.shape == (8,8): # General check for the 2x4 case
    #     print(f"\n  DEBUG INSIDE partial_channel (rho is {rho.shape}):")
    #     print(f"    Received sys argument: {sys} (expected 1 then 2 from is_separable BH loop)")
    #     print(f"    `dim` array at this point:\n{dim}")

    prod_dim_r1 = int(np.prod(dim[0, : sys - 1]))  # sys must be 1-indexed
    prod_dim_c1 = int(np.prod(dim[1, : sys - 1]))
    prod_dim_r2 = int(np.prod(dim[0, sys:]))  # sys is 1-indexed, slicing from index sys onwards
    prod_dim_c2 = int(np.prod(dim[1, sys:]))

    if isinstance(phi_map, list):
        # Compute the Kraus operators on the full system.
        s_phi_1, s_phi_2 = len(phi_map), len(phi_map[0])
        phi_list = []
        # Map is completely positive if input is given as:
        # 1. [K1, K2, .. Kr]
        # 2. [[K1], [K2], .. [Kr]]
        # 3. [[K1, K2, .. Kr]] and r > 2
        if isinstance(phi_map[0], np.ndarray):
            phi_list = phi_map
        elif s_phi_2 == 1 or s_phi_1 == 1 and s_phi_2 > 2:
            phi_list = list(itertools.chain(*phi_map))

        if phi_list:
            phi = []
            for m in phi_list:
                phi.append(
                    np.kron(
                        np.kron(np.identity(prod_dim_r1), m),
                        np.identity(prod_dim_r2),
                    )
                )
            phi_x = apply_channel(rho, phi)
        else:
            phi_1 = []
            for m in phi_map:
                phi_1.append(
                    np.kron(
                        np.kron(np.identity(prod_dim_r1), m[0]),
                        np.identity(prod_dim_r2),
                    )
                )
            phi_2 = []
            for m in phi_map:
                phi_2.append(
                    np.kron(
                        np.kron(np.identity(prod_dim_c1), m[1]),
                        np.identity(prod_dim_c2),
                    )
                )

            phi_x = [list(litem) for litem in zip(phi_1, phi_2)]
            phi_x = apply_channel(rho, phi_x)
        return phi_x

    # The `phi_map` variable is provided as a Choi matrix.
    if isinstance(phi_map, np.ndarray):
        dim_phi_input_map = phi_map.shape  # Shape of Choi for subsystem map

        # This is the dim array used for permute_systems later.
        # Let's call it dim_for_permutation_setup
        dim_for_permutation_setup = np.array(
            [
                [
                    prod_dim_r1,  # before_in
                    prod_dim_r1,  # before_out
                    int(dim[0, sys - 1]),  # subsys_in_dim_rows
                    int(dim_phi_input_map[0] / dim[0, sys - 1]),  # subsys_map_out_dim_rows
                    prod_dim_r2,  # after_in
                    prod_dim_r2,  # after_out
                ],
                [
                    prod_dim_c1,  # before_in_cols
                    prod_dim_c1,  # before_out_cols
                    int(dim[1, sys - 1]),  # subsys_in_dim_cols
                    int(dim_phi_input_map[1] / dim[1, sys - 1]),  # subsys_map_out_dim_cols
                    prod_dim_c2,  # after_in_cols
                    prod_dim_c2,  # after_out_cols
                ],
            ]
        )

        # These are Choi matrices of Identity maps on "before" and "after" spaces
        psi_r1 = max_entangled(prod_dim_r1, False, False)
        psi_c1 = max_entangled(prod_dim_c1, False, False)
        psi_r2 = max_entangled(prod_dim_r2, False, False)
        psi_c2 = max_entangled(prod_dim_c2, False, False)

        # X1 is J(Id_before), X2 is J(Id_after)
        # Assuming prod_dim_r1 == prod_dim_c1 and prod_dim_r2 == prod_dim_c2
        X1 = psi_r1 @ psi_c1.conj().T  # Shape (prod_dim_r1^2, prod_dim_c1^2)
        X2 = psi_r2 @ psi_c2.conj().T  # Shape (prod_dim_r2^2, prod_dim_c2^2)

        # This is J(Id_before) otimes J(Phi_sys) otimes J(Id_after)
        arg_for_permute = np.kron(np.kron(X1, phi_map), X2)

        # This permuted map should be J( Id_before otimes Phi_sys otimes Id_after )
        # It acts on the total space of rho. So its size should be rho.shape[0]^2 x rho.shape[0]^2
        phi_map_overall_choi = permute_systems(
            arg_for_permute,
            [0, 2, 4, 1, 3, 5],  # 0-indexed permutation
            dim_for_permutation_setup,  # This dim describes the 6-partite structure of arg_for_permute
        )

        # if is_debug_bh_2x4:
        #     print(f"\n  DEBUG INSIDE partial_channel (for 2x4 overall system,
        # map on actual subsystem index {sys-1}):")
        #     print(f"    Input rho shape: {rho.shape}")
        #     print(f"    Input phi_map_input (Choi for subsystem) shape: {phi_map.shape}")
        #     print(f"    X1 (J(Id_before)) shape: {X1.shape}")
        #     print(f"    X2 (J(Id_after)) shape: {X2.shape}")
        #     print(f"    arg_for_permute shape: {arg_for_permute.shape}")
        #     print(f"    dim_for_permutation_setup product (rows): {np.prod(dim_for_permutation_setup[0,:])}")
        #     print(f"    phi_map_overall_choi (output of permute_systems) shape: {phi_map_overall_choi.shape}")

        result_from_apply_channel = apply_channel(rho, phi_map_overall_choi)

        # if is_debug_bh_2x4:
        #     print(f"    Output from apply_channel (this is what partial_channel returns)
        # shape: {result_from_apply_channel.shape}")

        return result_from_apply_channel

    raise ValueError(
        "The `phi_map` variable is assumed to be provided as either a Choi matrix or a list of Kraus operators."
    )
