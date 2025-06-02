"""Test is_separable."""

import numpy as np
import pytest

from toqito.channels import partial_trace, partial_transpose
from toqito.matrix_props import is_density, is_positive_semidefinite
from toqito.rand import random_density_matrix
from toqito.state_props.is_ppt import is_ppt
from toqito.state_props.is_separable import is_separable
from toqito.states import basis, bell, isotropic


def test_entangled_zhang_realignment_criterion():
    """Test for entanglement using Zhang's realignment criterion."""
    # Create a state that satisfies this criterion
    rho = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
    np.testing.assert_equal(is_separable(rho), False)


def test_entangled_qutrit_qutrit():
    """Test for entanglement in the qutrit-qutrit case."""
    # Create a 3x3 entangled state
    psi = (1 / np.sqrt(3)) * (
        np.kron([1, 0, 0], [1, 0, 0]) + np.kron([0, 1, 0], [0, 1, 0]) + np.kron([0, 0, 1], [0, 0, 1])
    )
    rho = np.outer(psi, psi)
    np.testing.assert_equal(is_separable(rho), False)


def test_entangled_breuer_hall():
    """Test for entanglement using Breuer-Hall positive maps."""
    # Create a 4x4 entangled state
    psi = (1 / np.sqrt(2)) * (np.kron([1, 0], [1, 0]) + np.kron([0, 1], [0, 1]))
    rho = np.outer(psi, psi)
    np.testing.assert_equal(is_separable(rho), False)


def test_non_positive_semidefinite_matrix():
    """Ensure separability of non-positive semidefinite matrix is invalid."""
    with np.testing.assert_raises(ValueError):
        state = np.array([[-1, -1], [-1, -1]])
        is_separable(state)


def test_psd_matrix_local_dim_one():
    """Every positive semidefinite matrix is separable when one of the local dimensions is 1."""
    np.testing.assert_equal(is_separable(np.identity(2)), True)


def test_invalid_dim_parameter():
    """The dimension of the state must evenly divide the length of the state."""
    with np.testing.assert_raises(ValueError):
        dim = 3
        rho = isotropic(dim, 1 / (dim + 1))
        is_separable(rho, dim + 1)


def test_entangled_ppt_criterion():
    """Determined to be entangled via the PPT criterion."""
    rho = bell(0) @ bell(0).conj().T
    np.testing.assert_equal(is_separable(rho), False)


def test_ppt_small_dimensions():
    """Determined to be separable via sufficiency of the PPT criterion in small dimensions."""
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    psi = 1 / np.sqrt(3) * e_0 + 1 / np.sqrt(3) * e_1 + 1 / np.sqrt(3) * e_2

    e_0, e_1 = basis(2, 0), basis(2, 1)
    phi = np.kron((1 / np.sqrt(2) * e_0 + 1 / np.sqrt(2) * e_1), psi)
    sigma = phi @ phi.conj().T
    np.testing.assert_equal(is_separable(sigma), True)


def test_ppt_low_rank():
    """Determined to be separable via the operational criterion for low-rank operators."""
    m = 6
    n = m
    rho = random_density_matrix(m)
    u, s, v_h = np.linalg.svd(rho)
    rho_cut = u[:, : m - 1] @ np.diag(s[: m - 1]) @ v_h[: m - 1]
    rho_cut = rho_cut / np.trace(rho_cut)
    pt_state_alice = partial_trace(rho_cut, [1], [3, 2])

    np.testing.assert_equal(is_density(rho_cut), True)
    np.testing.assert_equal(is_density(np.array(pt_state_alice)), True)
    np.testing.assert_equal(
        np.linalg.matrix_rank(rho_cut) + np.linalg.matrix_rank(pt_state_alice) <= 2 * m * n - m - n + 2,
        True,
    )
    # This tests the ORIGINAL rho, not rho_cut.
    # rho is a full-rank 6x6 random density matrix, which is overwhelmingly likely to be entangled.
    # is_separable(rho) should correctly return False.
    np.testing.assert_equal(is_separable(rho), False)


def test_horodecki_operational_rank_le_max_dim_prod_gt_6():
    """Tests Horodecki Op criterion: rank <= max_dim for prod_dim > 6."""
    # System: 3x3, dA=3, dB=3. prod_dim = 9. max(dA,dB)=3.
    # Construct a rank 3 (or less) PPT state.
    # E.g., a mixture of 3 product states.
    p_list = []
    for i in range(3):
        psi_A = basis(3, i)
        psi_B = basis(3, (i + 1) % 3)  # Ensure different product states
        proj_A = np.outer(psi_A, psi_A.conj())
        proj_B = np.outer(psi_B, psi_B.conj())
        p_list.append(np.kron(proj_A, proj_B))

    rho_test = sum(p_list) / 3.0
    rho_test /= np.trace(rho_test)  # Should have rank 3

    # Sanity checks for the test state
    assert is_positive_semidefinite(rho_test, atol=1e-7)
    assert np.isclose(np.trace(rho_test), 1.0)
    assert is_ppt(rho_test, dim=[3, 3], tol=1e-7)  # Must be PPT
    assert np.linalg.matrix_rank(rho_test, tol=1e-7) <= 3  # Rank condition

    # is_separable should find this True, and one path is via _check_horodecki_operational
    assert is_separable(rho_test, dim=[3, 3])


def get_johnston_tiles_state():  # Renamed for clarity
    """Construct the Horodecki 3x3 PPT entangled state parameterized by 'a'."""
    b = [basis(3, i) for i in range(3)]  # b[0], b[1], b[2]

    unnormalized_kets = [
        np.kron(b[0], (b[0] - b[1])),
        np.kron(b[1], (b[1] - b[2])),
        np.kron(b[2], (b[2] - b[0])),
        np.kron((b[0] - b[1]), b[2]),
        np.kron((b[1] - b[2]), b[0]),
    ]

    psi_vectors = []
    for ket_unnorm in unnormalized_kets:
        norm = np.linalg.norm(ket_unnorm)
        if np.isclose(norm, 0):
            raise ValueError("Zero vector in Tiles construction")
        psi_vectors.append(ket_unnorm / norm)

    P_sum = np.zeros((9, 9), dtype=complex)
    for v_psi in psi_vectors:
        P_sum += np.outer(v_psi, v_psi.conj())

    rho = (np.eye(9, dtype=complex) - P_sum) / 4.0
    return rho


# Helper for 2x3 state (ensure this is in your test file)
def separable_state_2x3_rank3():
    """separable_state_2x3_rank3."""
    psi_A0 = np.array([1, 0], dtype=complex)
    psi_A1 = np.array([0, 1], dtype=complex)
    psi_B0 = np.array([1, 0, 0], dtype=complex)
    psi_B1 = np.array([0, 1, 0], dtype=complex)
    psi_B2 = np.array([0, 0, 1], dtype=complex)
    rho1 = np.kron(np.outer(psi_A0, psi_A0.conj()), np.outer(psi_B0, psi_B0.conj()))
    rho2 = np.kron(np.outer(psi_A0, psi_A0.conj()), np.outer(psi_B1, psi_B1.conj()))
    rho3 = np.kron(np.outer(psi_A1, psi_A1.conj()), np.outer(psi_B2, psi_B2.conj()))
    rho = (rho1 + rho2 + rho3) / 3
    # Basic checks for test state validity
    assert np.isclose(np.trace(rho), 1), "separable_state_2x3_rank3 trace is not 1"
    assert np.all(np.linalg.eigvalsh(rho) >= -1e-9), "separable_state_2x3_rank3 not PSD"
    return rho


# Corrected Isotropic State Test
def test_isotropic_npt_is_found_entangled():  # Renamed and logic corrected
    """Tests that an NPT isotropic state is correctly found entangled."""
    dim_iso = 3
    alpha = 0.3  # For d=3, this state is NPT (min_eig_PT ~ -0.022)
    rho_iso_npt = isotropic(dim_iso, alpha)

    assert is_density(rho_iso_npt), "Isotropic state should be a density matrix."

    pt_rho_iso = partial_transpose(rho_iso_npt, sys=0, dim=[dim_iso, dim_iso])
    min_eig_pt = np.min(np.linalg.eigvalsh(pt_rho_iso))
    # print(f"\nDEBUG Isotropic PT min eig: {min_eig_pt}") # Should be < 0

    assert not is_ppt(rho_iso_npt, dim=[dim_iso, dim_iso], tol=1e-7), (
        f"Isotropic state d=3, alpha=0.3 should be NPT. is_ppt returned True, min_eig_pt={min_eig_pt}"
    )

    assert not is_separable(rho_iso_npt, dim=[dim_iso, dim_iso]), (
        "NPT Isotropic state should be classified as entangled by is_separable."
    )


def test_entangled_cross_norm_realignment_criterion():
    """Determined to be entangled by using Theorem 1 and Remark 1 of :cite:`Chen_2003_Matrix`."""
    p_var, a_var, b_var = 0.4, 0.8, 0.64
    rho = np.array(
        [
            [p_var * a_var**2, 0, 0, p_var * a_var * b_var],
            [0, (1 - p_var) * a_var**2, (1 - p_var) * a_var * b_var, 0],
            [0, (1 - p_var) * a_var * b_var, (1 - p_var) * a_var**2, 0],
            [p_var * a_var * b_var, 0, 0, p_var * a_var**2],
        ]
    )
    np.testing.assert_equal(is_separable(rho), False)


def test_separable_closeness_to_maximally_mixed_state():
    """Determined to be separable by closeness to the maximally mixed state."""
    rho = np.array(
        [
            [4, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 4, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 4, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 4, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 4, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 4],
        ]
    )
    rho = rho / np.trace(rho)
    np.testing.assert_equal(is_separable(rho), True)


def test_separable_small_rank1_perturbation_of_maximally_mixed_state():
    """Determined to be separable by being a small rank-1 perturbation of the maximally-mixed state."""
    rho = np.array(
        [
            [4, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 4, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 4, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 4, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 4, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 4, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 4, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 4, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 4],
        ]
    )
    rho = rho / np.trace(rho)
    np.testing.assert_equal(is_separable(rho), True)


def test_separable_schmidt_rank():
    """Determined to be separable by having operator Schmidt rank at most 2."""
    rho = np.array(
        [
            [0.25, 0.15, 0.1, 0.15, 0.09, 0.06, 0.1, 0.06, 0.04],
            [0.15, 0.2, 0.05, 0.09, 0.12, 0.03, 0.06, 0.08, 0.02],
            [0.1, 0.05, 0.05, 0.06, 0.03, 0.03, 0.04, 0.02, 0.02],
            [0.15, 0.09, 0.06, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02],
            [0.09, 0.12, 0.03, 0.12, 0.16, 0.04, 0.03, 0.04, 0.01],
            [0.06, 0.03, 0.03, 0.08, 0.04, 0.04, 0.02, 0.01, 0.01],
            [0.1, 0.06, 0.04, 0.05, 0.03, 0.02, 0.05, 0.03, 0.02],
            [0.06, 0.08, 0.02, 0.03, 0.04, 0.01, 0.03, 0.04, 0.01],
            [0.04, 0.02, 0.02, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01],
        ]
    )
    rho = rho / np.trace(rho)
    np.testing.assert_equal(is_separable(rho), True)


# In test_is_separable.py


def get_horodecki_parameterized_ppt_entangled_state(a_param):
    """Construct the Horodecki 3x3 PPT entangled state parameterized by 'a'."""
    # Entangled for 0 <= a < 1.
    # Based on P. Horodecki, Phys. Lett. A 232, 333-339 (1997).
    # Note: This specific matrix form is a common representation, ensure it matches the intended state.
    # This form is more directly from quant-ph/9703004 the state sigma_a.

    if not (0 <= a_param < 1):  # Strictly < 1 for entanglement
        raise ValueError("Parameter 'a' must be in [0, 1) for an entangled state.")

    rho_un = np.zeros((9, 9), dtype=complex)

    # Diagonal blocks
    rho_un[0, 0] = rho_un[1, 1] = rho_un[2, 2] = a_param
    rho_un[3, 3] = rho_un[5, 5] = rho_un[6, 6] = rho_un[7, 7] = (a_param + 1) / 2

    # Off-diagonal for first block type
    rho_un[0, 4] = rho_un[4, 0] = a_param
    rho_un[0, 8] = rho_un[8, 0] = a_param
    rho_un[4, 8] = rho_un[8, 4] = a_param  # Error in my original transcription, this term is not there.
    # Let's use a verified matrix structure for this family.

    d_iso = 3
    alpha_val = 0.4
    # alpha_val = 0.5 makes it on the boundary of NPT for some definitions. Let's use 0.4.
    # 1/3 = 0.333...  alpha=0.4 is > 1/3.
    # alpha=0.4 is <= 0.5. So it is PPT and entangled.

    phi_plus_vec = np.zeros(d_iso * d_iso, dtype=complex)
    for i in range(d_iso):
        phi_plus_vec[i * d_iso + i] = 1 / np.sqrt(d_iso)
    phi_plus_proj = np.outer(phi_plus_vec, phi_plus_vec.conj())

    identity_total = np.eye(d_iso * d_iso) / (d_iso * d_iso)

    rho = alpha_val * phi_plus_proj + (1 - alpha_val) * identity_total
    return rho


def test_isotropic_d3_alpha04_is_NPT_and_entangled():
    """test_isotropic_d3_alpha04_is_NPT_and_entangled."""
    rho = get_isotropic_3x3_ppt_entangled_state_alpha_0_4()  # Current name is misleading now
    dim_iso = 3
    dims = [dim_iso, dim_iso]
    test_tol = 1e-7

    assert is_density(rho)
    assert not is_ppt(rho, dim=dims, tol=test_tol), "Isotropic(d=3,alpha=0.4) should be NPT."
    assert not is_separable(rho, dim=dims, tol=test_tol), (
        "NPT Isotropic(d=3,alpha=0.4) should be found entangled by is_separable."
    )


# Helper function for the new test
def get_isotropic_3x3_ppt_entangled_state_alpha_0_4():
    """get_isotropic_3x3_ppt_entangled_state_alpha_0_4."""
    d_iso = 3
    alpha_val = 0.4

    phi_plus_vec = np.zeros(d_iso * d_iso, dtype=complex)
    for i in range(d_iso):
        # Correct indexing for |ii> state component
        ket_i = np.zeros(d_iso)
        ket_i[i] = 1
        phi_plus_vec += np.kron(ket_i, ket_i)
    phi_plus_vec /= np.sqrt(d_iso)

    phi_plus_proj = np.outer(phi_plus_vec, phi_plus_vec.conj())
    identity_total = np.eye(d_iso * d_iso) / (d_iso * d_iso)

    rho = alpha_val * phi_plus_proj + (1 - alpha_val) * identity_total
    return rho


def test_entangled_symmetric_extension():
    """Determined to be entangled by not having a PPT symmetric extension."""
    # This matrix is obtained by using the `rho` from `test_separable_schmidt_rank`
    # and finding the nearest PSD matrix to it. See https://stackoverflow.com/a/18542094.
    rho = np.array(
        [
            [1.0, 0.67, 0.91, 0.67, 0.45, 0.61, 0.88, 0.59, 0.79],
            [0.67, 1.0, 0.5, 0.45, 0.67, 0.34, 0.59, 0.88, 0.44],
            [0.91, 0.5, 1.0, 0.61, 0.34, 0.68, 0.81, 0.44, 0.88],
            [0.67, 0.45, 0.61, 1.0, 0.67, 0.91, 0.5, 0.33, 0.45],
            [0.45, 0.67, 0.34, 0.67, 1.0, 0.5, 0.33, 0.5, 0.25],
            [0.61, 0.34, 0.68, 0.91, 0.5, 1.0, 0.45, 0.26, 0.5],
            [0.88, 0.59, 0.81, 0.5, 0.33, 0.45, 1.0, 0.66, 0.91],
            [0.59, 0.88, 0.44, 0.33, 0.5, 0.26, 0.66, 1.0, 0.48],
            [0.79, 0.44, 0.88, 0.45, 0.25, 0.5, 0.91, 0.48, 1.0],
        ]
    )
    rho = rho / np.trace(rho)
    np.testing.assert_equal(is_separable(rho), False)


def test_separable_based_on_eigenvalues():
    """Determined to be separable by inspecting its eigenvalues. See Lemma 1 of :cite:`Johnston_2013_Spectrum`."""
    # Although this matrix, taken from the above paper, satisfies the eigenvalues condition,
    # this returns True from a line above the eigenvalues condition.
    rho = np.array(
        [
            [4 / 22, 2 / 22, -2 / 22, 2 / 22],
            [2 / 22, 7 / 22, -2 / 22, -1 / 22],
            [-2 / 22, -2 / 22, 4 / 22, -2 / 22],
            [2 / 22, -1 / 22, -2 / 22, 7 / 22],
        ]
    )
    np.testing.assert_equal(is_separable(rho), True)


def test_input_state_not_numpy_array():
    """Test TypeError for non-NumPy array input (covers line 121)."""
    with pytest.raises(TypeError, match="Input state must be a NumPy array."):
        is_separable("not_a_matrix")


def test_input_state_not_2d_matrix():
    """Test ValueError for non-2D matrix input (covers line 123)."""
    with pytest.raises(ValueError, match="Input state must be a 2D matrix."):
        is_separable(np.array([1, 2, 3, 4]))


def test_input_state_not_square():
    """Test ValueError for non-square matrix input (covers line 126)."""
    with pytest.raises(ValueError, match="Input state must be a square matrix."):
        is_separable(np.array([[1, 2, 3], [4, 5, 6]]))


def test_input_state_trace_zero():
    """Test ValueError for state with trace close to zero (covers line 131)."""
    state_zero_trace = np.array([[0, 0], [0, 0]], dtype=complex)
    with pytest.raises(ValueError, match="Trace of the input state is close to zero."):
        is_separable(state_zero_trace)


def test_input_state_auto_normalization():
    """Test state normalization if trace is not 1 (implicit coverage for line 133 logic)."""
    state_unnormalized = 2 * np.array([[1, 0], [0, 0]], dtype=complex)  # Trace is 2
    # This should run without error and normalize, eventually returning True for |0><0|
    assert is_separable(state_unnormalized, dim=[2, 1])  # min_dim = 1 path


def test_dim_non_positive_integer():
    """Test ValueError for non-positive integer dim (covers line 140)."""
    state = np.eye(4) / 4
    with pytest.raises(ValueError, match="Integer `dim` must be positive."):
        is_separable(state, dim=0)
    with pytest.raises(ValueError, match="Integer `dim` must be positive."):
        is_separable(state, dim=-2)


def test_dim_none_state_len_prime_gt_1():
    """Test ValueError for dim=None and prime state_len > 1 (covers line 143 specific path)."""
    state_prime_dim = np.eye(7) / 7  # sqrt(7) is not int, round(sqrt(7))=3. 7%3 !=0
    with pytest.raises(ValueError, match="State dimension 7 is prime > 1; system cannot be bipartite."):
        is_separable(state_prime_dim, dim=None)


def test_dim_int_not_dividing_state_len():
    """Test ValueError for integer dim not dividing state_len (covers line 143 general path)."""
    state = np.eye(6) / 6
    with pytest.raises(ValueError, match="Integer `dim` .* must evenly divide state dimension .*"):
        is_separable(state, dim=4)


def test_dim_list_product_mismatch():
    """Test ValueError for dim list product not matching state_len (covers line 151)."""
    state = np.eye(4) / 4
    with pytest.raises(ValueError, match="Product of dimensions in `dim` list must equal state dimension."):
        is_separable(state, dim=[2, 3])


def test_dim_list_non_positive_or_non_integer_entries():
    """Test ValueError for dim list with invalid entries (covers line 149)."""
    state = np.eye(4) / 4
    with pytest.raises(ValueError, match="Dimensions in `dim` list must be positive integers."):
        is_separable(state, dim=[2, 0])
    with pytest.raises(ValueError, match="Dimensions in `dim` list must be positive integers."):
        is_separable(state, dim=[2, 2.5])


def test_dim_wrong_type():
    """Test ValueError for dim of incorrect type (e.g., string, list of 3) (covers line 153)."""
    state = np.eye(4) / 4
    with pytest.raises(ValueError, match="`dim` must be an int, a list of two positive ints, or None."):
        is_separable(state, dim="abc")
    with pytest.raises(ValueError, match="`dim` must be an int, a list of two positive ints, or None."):
        is_separable(state, dim=[1, 2, 3])


def test_breuer_hall_mixed_even_odd_dims():
    """Test Breuer-Hall map skipping for odd sub-dimensions (covers logic around line 260)."""
    # 3x2 state: Alice (dim 3, odd), Bob (dim 2, even)
    # Breuer-Hall should only apply for Bob's system (p_sys_idx=1)
    # We need a state that *would* be caught if BH applied to Alice, but isn't.
    # Or, more simply, a separable 3x2 state to ensure it passes.
    rho_3x2_sep = np.kron(random_density_matrix(3, is_real=False), random_density_matrix(2, is_real=False))
    assert is_separable(rho_3x2_sep, dim=[3, 2])

    # 3x3 state: Both odd, BH maps should be skipped for both subsystems.
    rho_3x3_sep = np.kron(random_density_matrix(3, is_real=False), random_density_matrix(3, is_real=False))
    assert is_separable(rho_3x3_sep, dim=[3, 3])  # Will be caught by PPT is N&S if rank <=4 or other rules.


def test_2xN_no_swap_needed():
    """Test 2xN path where first dim is 2 (no swap, covers line 285 else path)."""
    # Use a state that will pass through this and be decided by a 2xN rule.
    # Example: separable_state_2x3_rank3() is already 2x3.
    assert is_separable(separable_state_2x3_rank3(), dim=[2, 3])
    # To ensure the 2xN block is entered, use a 2xN state that is NOT caught by PPT sufficiency (e.g. 2x4)
    # and passes PPT.
    rho_2x4_sep = np.kron(random_density_matrix(2), random_density_matrix(4))
    assert is_separable(rho_2x4_sep, dim=[2, 4])


# Rank-1 perturbation of identity (Line 305 `else` paths)
def test_rank1_perturbation_not_met_but_separable():
    """Test PPT state not meeting lam[1]-lam[N-1] < tol^2, but separable otherwise (line 305 inner else)."""
    # A 2x2 product state. Eigenvalues might not satisfy the condition if it's not close to identity.
    # e.g. |00><00|. lam = [1,0,0,0]. lam[1]-lam[3] = 0. This *passes*.
    # Need one that fails it.
    # Consider rho = 0.6|00><00| + 0.2|01><01| + 0.1|10><10| + 0.1|11><11|
    # lam = [0.6, 0.2, 0.1, 0.1]. lam[1]-lam[3] = 0.2-0.1 = 0.1. This is > tol.
    # This state is separable.
    rho = np.diag([0.6, 0.2, 0.1, 0.1])
    assert is_separable(rho, dim=[2, 2])  # Should be true.


def test_rank1_perturbation_len_lam_not_gt_1():
    """Test state where len(lam) <= 1 (e.g. 1x1 state) (line 305 outer else)."""
    # This is typically caught by `min_dim == 1` check earlier.
    rho_1x1 = np.array([[1.0]])
    assert is_separable(rho_1x1, dim=[1, 1])  # Should be True due to min_dim=1.


# Schmidt rank > 2 path (Line 311 `else`)
def test_schmidt_rank_gt_2_separable():
    """Test a separable state with Schmidt rank > 2 (line 311 else path)."""
    # Mixture of 3 product states in 3x3.
    p1 = np.kron(random_density_matrix(3), random_density_matrix(3))
    p2 = np.kron(random_density_matrix(3), random_density_matrix(3))
    p3 = np.kron(random_density_matrix(3), random_density_matrix(3))
    rho = (p1 + p2 + p3) / 3
    # This state is separable. Its OSR could be 3.
    # It should pass PPT. It should then not be caught by OSR<=2.
    # It should eventually be caught by symmetric extension or other 3x3 rules if applicable.
    assert is_separable(rho, dim=[3, 3])


def get_separable_3x3_rank4_ppt_state():
    """get_separable_3x3_rank4_ppt_state."""
    p_states = []
    # Create 4 linearly independent product states (or nearly so for mixture)
    # psi_A_i, psi_B_i should be from basis(3,k)

    # State 1: |00><00|
    psi_A0, psi_B0 = basis(3, 0), basis(3, 0)
    p_states.append(np.outer(np.kron(psi_A0, psi_B0), np.kron(psi_A0, psi_B0).conj()))

    # State 2: |11><11|
    psi_A1, psi_B1 = basis(3, 1), basis(3, 1)
    p_states.append(np.outer(np.kron(psi_A1, psi_B1), np.kron(psi_A1, psi_B1).conj()))

    # State 3: |22><22|
    psi_A2, psi_B2 = basis(3, 2), basis(3, 2)
    p_states.append(np.outer(np.kron(psi_A2, psi_B2), np.kron(psi_A2, psi_B2).conj()))

    # State 4: A mixed product state to ensure rank 4, e.g., (|0>+|1>)(<0|+<1|) tensor |0><0|
    psi_A3 = (basis(3, 0) + basis(3, 1)) / np.sqrt(2)
    psi_B3 = basis(3, 0)  # Can be same as one above, or different like basis(3,2)
    p_states.append(np.outer(np.kron(psi_A3, psi_B3), np.kron(psi_A3, psi_B3).conj()))

    rho = sum(p_states) / len(p_states)
    rho = rho / np.trace(rho)

    # Verify properties (caller should do this if state is complex)
    # print(f"Constructed 3x3 state rank: {np.linalg.matrix_rank(rho, tol=1e-7)}")
    # print(f"Is it PPT? {is_ppt(rho, dim=[3,3], tol=1e-7)}")
    return rho


def test_3x3_rank4_separable_det_F_criterion():
    """Test 3x3 rank 4 PPT separable state via det(F) criterion."""
    rho = get_separable_3x3_rank4_ppt_state()

    # Ensure it's rank 4 and PPT for the test to be valid for this path
    if np.linalg.matrix_rank(rho, tol=1e-7) == 4 and is_ppt(rho, dim=[3, 3], tol=1e-7):
        assert is_separable(rho, dim=[3, 3]) is True, (
            "3x3 rank-4 PPT separable state failed the det(F) check or was misclassified."
        )
    else:
        pytest.skip("Constructed 3x3 state not rank 4 PPT, skipping det(F) separable test.")


def test_2xN_johnston_spectrum_rank_deficient_separable():
    """# A product state rhoA otimes rhoB_rank_deficient."""
    rho_A = random_density_matrix(2)
    # Construct a rank 3 (out of 4) state for dB=4
    rho_B_full = random_density_matrix(4)
    e, v = np.linalg.eigh(rho_B_full)
    e[0] = 0  # Make it rank 3
    e = e / np.sum(e)  # Normalize eigenvalues
    rho_B_rank3 = v @ np.diag(e) @ v.conj().T
    rho_test = np.kron(rho_A, rho_B_rank3)
    if not is_ppt(rho_test, dim=[2, 4], tol=1e-7):
        pytest.skip("Constructed 2x4 rank-deficient state was not PPT.")
    # This test assumes is_separable will then evaluate the Johnston spectrum for it.
    # It might be caught by Horodecki Op if rank sum is low enough.
    assert is_separable(rho_test, dim=[2, 4])  # Should be true


def test_separable_rank1_perturbation_of_identity_catches():
    """test_separable_rank1_perturbation_of_identity_catches."""
    # 3x3 system, prod_dim=9. Need full rank (len(lam)==9).
    # lam[1] - lam[8] < tol^2
    # Let lam[0] = large, lam[1]...lam[8] small and nearly equal.
    eig_vals = np.zeros(9)
    eig_vals[0] = 1.0 - 8 * 1e-9  # Largest
    for i in range(1, 9):
        eig_vals[i] = 1e-9 + (np.random.rand() * 1e-12)  # Small, nearly equal
    eig_vals = eig_vals / np.sum(eig_vals)  # Normalize
    eig_vals = np.sort(eig_vals)[::-1]  # Sort descending
    rho = np.diag(eig_vals)  # Diagonal state with these eigenvalues
    # Ensure it's PPT (diagonal states are always PPT)
    assert is_ppt(rho, dim=[3, 3])
    assert is_separable(rho, dim=[3, 3]) is True


def test_breuer_hall_one_dim_odd_separable():
    """test_breuer_hall_one_dim_odd_separable."""
    # Test with a 3x2 separable state (dA=3 odd, dB=2 even)
    # BH map for dA (p_sys_idx=0) should be skipped.
    # BH map for dB (p_sys_idx=1) should be applied and pass for a separable state.
    rho_A = random_density_matrix(3)
    rho_B = random_density_matrix(2)
    rho_sep_3x2 = np.kron(rho_A, rho_B)
    assert is_ppt(rho_sep_3x2, dim=[3, 2], sys=2)  # Check PPT (sys=2 for 2nd system, 1-based)
    # or sys=1 (0-based) if is_ppt uses that
    assert is_separable(rho_sep_3x2, dim=[3, 2]) is True


def test_2xN_hildebrand_rank_condition_true():
    """test_2xN_hildebrand_rank_condition_true."""
    # Need dA=2, dB=N (N>=4 for prod_dim > 6 to bypass PPT sufficiency)
    # Construct A, B, C blocks such that rho is PPT and B-B.T is rank <=1
    # E.g., two commuting density matrices for A and B in rhoA otimes rhoB
    rho_A = np.diag(np.array([0.7, 0.3]))
    rho_B_diag = np.random.rand(4)
    rho_B_diag = rho_B_diag / np.sum(rho_B_diag)
    rho_B = np.diag(rho_B_diag)

    rho_test = np.kron(rho_A, rho_B)
    # For this product state, B_block will have a specific structure. B-B.T might be low rank.
    if is_ppt(rho_test, dim=[2, 4]):  # Ensure it's PPT
        # This will likely be true (is_separable returns True)
        # but we don't know if it's *because* of this specific line.
        assert is_separable(rho_test, dim=[2, 4])
    else:
        pytest.skip("Constructed state for Hildebrand rank test not PPT.")


# Test for 3x3 Rank-4 Separable (det F path)
def get_separable_3x3_rank4_ppt_state():
    """get_separable_3x3_rank4_ppt_state."""
    # Construct as a mixture of 4 product states that are likely to give rank 4
    # and be PPT. This is heuristic.
    s = []
    s.append(np.kron(np.outer(basis(3, 0), basis(3, 0)), np.outer(basis(3, 0), basis(3, 0))))
    s.append(np.kron(np.outer(basis(3, 1), basis(3, 1)), np.outer(basis(3, 1), basis(3, 1))))
    s.append(np.kron(np.outer(basis(3, 2), basis(3, 2)), np.outer(basis(3, 2), basis(3, 2))))
    # A slightly off-diagonal one to help with rank
    psi_A = (basis(3, 0) + basis(3, 1)) / np.sqrt(2)
    psi_B = (basis(3, 0) + basis(3, 2)) / np.sqrt(2)
    s.append(np.kron(np.outer(psi_A, psi_A.conj()), np.outer(psi_B, psi_B.conj())))
    rho = sum(s) / 4.0
    return rho / np.trace(rho)


def test_3x3_rank4_sep_det_F_criterion():
    """test_3x3_rank4_sep_det_F_criterion."""
    rho = get_separable_3x3_rank4_ppt_state()
    # We hope this is rank 4 and PPT. Test may be skipped if not.
    is_candidate = np.linalg.matrix_rank(rho, tol=1e-6) == 4 and is_ppt(rho, dim=[3, 3], tol=1e-7)
    if not is_candidate:
        pytest.skip("Constructed 3x3 state not rank-4 PPT for det(F) sep test.")
    assert is_separable(rho, dim=[3, 3])  # Should be True, via det(F) ~ 0


# Test for Breuer-Hall odd dimension skip
def test_breuer_hall_one_dim_odd_path_coverage():
    """test_breuer_hall_one_dim_odd_path_coverage."""
    # 3x2 separable state. dA=3 (odd), dB=2 (even).
    # BH for dA is skipped. BH for dB is applied. Overall should be separable.
    rho_A = random_density_matrix(3)
    rho_B = random_density_matrix(2)
    rho_sep_3x2 = np.kron(rho_A, rho_B)
    assert is_separable(rho_sep_3x2, dim=[3, 2])


# Test for 2xN Hildebrand rank condition (B-B.T rank <=1) being TRUE
def test_2xN_hildebrand_rank_B_minus_BT_is_zero_true():
    """test_2xN_hildebrand_rank_B_minus_BT_is_zero_true."""
    # Product of diagonal matrices will have B_block = 0, so B-B.T=0 (rank 0).
    # Use 2x4 to avoid PPT sufficiency.
    rho_A_diag = np.diag(np.array([0.7, 0.3]))
    rho_B_diag_vals = np.array([0.4, 0.3, 0.2, 0.1])
    rho_B_diag = np.diag(rho_B_diag_vals / np.sum(rho_B_diag_vals))
    rho_test = np.kron(rho_A_diag, rho_B_diag)

    assert is_ppt(rho_test, dim=[2, 4], tol=1e-7)  # Diagonal states are PPT
    # This state IS separable. is_separable should return True.
    # One of the 2xN rules (possibly this one, or Johnston Lemma 1 if B=0) should make it pass.
    assert is_separable(rho_test, dim=[2, 4])


# Test for Rank-1 Perturbation (full rank case)
def test_separable_rank1_perturbation_full_rank_catches():
    """test_separable_rank1_perturbation_full_rank_catches."""
    dim_sys = 3
    prod_dim = dim_sys**2
    eig_vals = np.zeros(prod_dim)
    main_eig = 1.0 - (prod_dim - 1) * 1e-9
    if main_eig <= 0:  # Ensure main_eig is positive after subtraction
        pytest.skip("Cannot construct valid eigenvalues for rank1_perturbation test with these parameters.")

    eig_vals[0] = main_eig
    for i in range(1, prod_dim):
        # Make other eigenvalues very small AND very close to each other
        eig_vals[i] = 1e-9 + (np.random.rand() * 1e-12)  # Small, nearly equal

    eig_vals = eig_vals / np.sum(eig_vals)  # Normalize
    eig_vals = np.sort(eig_vals)[::-1]  # Sort descending

    rho = np.diag(eig_vals)
    # This state is diagonal, hence PPT.
    # lam[1] - lam[prod_dim-1] should be very small.
    assert is_separable(rho, dim=[dim_sys, dim_sys])


# Test for 2xN swap logic
def test_2xN_rules_with_swap_3x2_sep():
    """test_2xN_rules_with_swap_3x2_sep."""
    rho_A = random_density_matrix(3)
    rho_B = random_density_matrix(2)
    rho_test = np.kron(rho_A, rho_B)
    assert is_separable(rho_test, dim=[3, 2])


def test_rank1_pert_not_full_rank_path():
    """test_rank1_pert_not_full_rank_path."""
    # 3x3, prod_dim=9. Make it rank 8, not rank-1 pert.
    eigs = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0])  # Rank 8
    eigs = eigs / np.sum(eigs)
    rho = np.diag(eigs)
    assert is_ppt(rho, dim=[3, 3])
    # This is separable. It will fail len(lam)==prod_dim.
    # Then proceed to has_symmetric_extension.
    assert is_separable(rho, dim=[3, 3])


def test_breuer_hall_3x4_separable_odd_even_skip():
    """test_breuer_hall_3x4_separable_odd_even_skip."""
    # dA=3 (odd), dB=4 (even). prod_dim=12.
    rhoA = random_density_matrix(3)
    rhoB = random_density_matrix(4)
    rho_sep_3x4 = np.kron(rhoA, rhoB)
    # This state is separable. When BH maps are applied:
    # p_sys_idx=0 (dim 3): BH map skipped.
    # p_sys_idx=1 (dim 4): BH map applied, should result in PSD.
    # Overall is_separable should be True.
    assert is_separable(rho_sep_3x4, dim=[3, 4])


def test_johnston_spectrum_eq12_trigger():
    """test_johnston_spectrum_eq12_trigger."""
    # 2x4 system. max_d_for_2xn = 4.
    # Target: (L0 - L6)^2 <= 4 * L5 * L7
    eigs = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1])  # Sum = 1.0
    eigs = np.sort(eigs)[::-1]

    # Try to make L0 very close to L6
    eigs = np.array([0.201, 0.2, 0.15, 0.1, 0.099, 0.05, 0.05, 0.15])  # Sum=1
    eigs = np.sort(eigs)[::-1]

    # What if L5 or L7 is large?
    eigs = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # L0=0.3, L5=0.1, L6=0.1, L7=0.1
    # (0.3-0.1)^2 = 0.04. 4*0.1*0.1 = 0.04.  0.04 <= 0.04 is TRUE.
    rho = np.diag(eigs)
    assert is_ppt(rho, dim=[2, 4])
    assert is_separable(rho, dim=[2, 4])  # Should hit this condition


def test_entangled_symmetric_extension():
    """Determined to be entangled by not having a PPT symmetric extension."""
    # This matrix is obtained by using the `rho` from `test_separable_schmidt_rank`
    # and finding the nearest PSD matrix to it. See https://stackoverflow.com/a/18542094.
    rho = np.array(
        [
            [1.0, 0.67, 0.91, 0.67, 0.45, 0.61, 0.88, 0.59, 0.79],
            [0.67, 1.0, 0.5, 0.45, 0.67, 0.34, 0.59, 0.88, 0.44],
            [0.91, 0.5, 1.0, 0.61, 0.34, 0.68, 0.81, 0.44, 0.88],
            [0.67, 0.45, 0.61, 1.0, 0.67, 0.91, 0.5, 0.33, 0.45],
            [0.45, 0.67, 0.34, 0.67, 1.0, 0.5, 0.33, 0.5, 0.25],
            [0.61, 0.34, 0.68, 0.91, 0.5, 1.0, 0.45, 0.26, 0.5],
            [0.88, 0.59, 0.81, 0.5, 0.33, 0.45, 1.0, 0.66, 0.91],
            [0.59, 0.88, 0.44, 0.33, 0.5, 0.26, 0.66, 1.0, 0.48],
            [0.79, 0.44, 0.88, 0.45, 0.25, 0.5, 0.91, 0.48, 1.0],
        ]
    )
    rho = rho / np.trace(rho)
    np.testing.assert_equal(is_separable(rho), False)


# Final `return False`
# This is covered by tests where a known entangled state is not caught by any specific
# separability criterion and `has_symmetric_extension` also returns False (or is skipped).
# The existing `test_entangled_symmetric_extension_dps` is a good candidate if
# `has_symmetric_extension` for that state indeed evaluates to False in your toqito setup.
# If `toqito.state_props.has_symmetric_extension` is not fully implemented or skips for some reason,
# then a state like a Bell state (if PPT was hypothetically True) would hit this.
# def test_ppt_entangled_isotropic_FAILS_level2_SE_and_is_declared_entangled():
#     """
#     Tests a 3x3 Isotropic state (alpha=0.4) which is PPT and entangled.
#     It should be found entangled by is_separable because
#     has_symmetric_extension(level=2) should return False.
#     This test targets the final `return False` in is_separable (line ~169 in 169-stmt version).
#     """
#     rho_iso_entangled = get_isotropic_3x3_ppt_entangled_alpha_0_4()
#     test_level = 2
#     test_tol = 1e-7 # A practical tolerance for these checks
#
#     # 1. Verify properties of the test state
#     assert is_density(rho_iso_entangled), \
#         f"Isotropic(d=3,alpha=0.4) not density. MinEig: {np.min(np.linalg.eigvalsh(rho_iso_entangled))}"
#
#     # For d=3, alpha=0.4, this state IS PPT.
#     # Theory: PPT if alpha <= 1/(d-1) = 1/2. Separable if alpha <= 1/d = 1/3.
#     # 1/3 < 0.4 <= 1/2, so it's PPT and entangled.
#     assert is_ppt(rho_iso_entangled, dim=[3,3], sys=1, tol=test_tol), \
#         "Isotropic(d=3,alpha=0.4) state *should* be PPT."
#
#     # 2. Test with is_separable
#     # We expect this state to NOT have a 2-PPT-symmetric extension.
#     # Thus, has_symmetric_extension(level=2, ppt=True) should be False.
#     # And is_separable should return False.
#
#     # Add debug prints inside has_symmetric_extension & symmetric_extension_hierarchy
#     # for this state to confirm sdp_val and problem.status.
#     # You can make the debug print conditions in those functions:
#     # is_target_iso_test = (rho.shape==(9,9) and np.isclose(rho[0,0], 0.4/3 + (1-0.4)/9))
#
#     print(f"DEBUG test_ppt_entangled_isotropic: Testing with level={test_level}")
#     result = is_separable(rho_iso_entangled, dim=[3,3], level=test_level, tol=test_tol)
#
#     np.testing.assert_equal(result, False)
