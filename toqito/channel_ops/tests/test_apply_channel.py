"""Tests for apply_channel."""

import numpy as np
import pytest

from toqito.channel_ops import apply_channel
from toqito.matrices import pauli
from toqito.perms import swap_operator
from toqito.states import basis

kraus_1 = np.array([[1, 5], [1, 0], [0, 2]])
kraus_2 = np.array([[0, 1], [2, 3], [4, 5]])
kraus_3 = np.array([[-1, 0], [0, 0], [0, -1]])
kraus_4 = np.array([[0, 0], [1, 1], [0, 0]])


@pytest.mark.parametrize(
    "input_mat, expected_result, apply_channel_arg2",
    [
        # The swap operator is the Choi matrix of the transpose map.
        # The following test is a (non-ideal, but illustrative) way of computing the transpose of a matrix.
        (np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), swap_operator(3)),
        # The swap operator is the Choi matrix of the transpose map.
        # The following test is a (non-ideal, but illustrative) way of computing the transpose of a non-square matrix.
        (np.array([[0, 1], [2, 3], [4, 5]]), np.array([[0, 2, 4], [1, 3, 5]]), swap_operator([2, 3])),
        # Apply Kraus map.
        # The following test computes PHI(X) where X = [[1, 2], [3, 4]] and where PHI is the superoperator defined by:
        # Phi(X) = [[1,5],[1,0],[0,2]] X [[0,1][2,3][4,5]].conj().T -[[1,0],[0,0],[0,1]] X [[0,0][1,1],[0,0]].conj().T
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[22, 95, 174], [2, 8, 14], [8, 29, 64]]),
            [[kraus_1, kraus_2], [kraus_3, kraus_4]],
        ),
    ],
)
def test_apply_channel(input_mat, expected_result, apply_channel_arg2):
    """Test function works as expected for valid inputs."""
    calculated_result = apply_channel(input_mat, apply_channel_arg2)
    assert (calculated_result == expected_result).all()


@pytest.mark.parametrize("nested", [1, 2, 3])
def test_apply_channel_cpt_kraus(nested):
    """Apply Kraus map of single qubit depolarizing channel."""
    test_input_mat = np.array([[1, 0], [0, 0]])

    expected_res = np.array([[0.5, 0], [0, 0.5]])

    kraus = [0.5 * pauli(ind) for ind in range(4)]
    if nested == 2:
        kraus = [kraus]
    elif nested == 3:
        kraus = [[mat] for mat in kraus]

    res = apply_channel(test_input_mat, kraus)
    assert (res == expected_res).all()


def test_apply_channel_invalid_input():
    """Invalid input for apply map."""
    with pytest.raises(ValueError):
        apply_channel(np.array([[1, 2], [3, 4]]), 2)


def test_apply_identity_channel_via_choi():
    """Test applying identity channel using its Choi matrix."""
    d = 2
    # Input matrix X
    x_vec = basis(d, 0)  # |0>
    x_op = np.outer(x_vec, x_vec.conj())  # |0><0|

    # Choi matrix of identity channel on C^d: J(Id) = sum_{i,j} |i><j| otimes |i><j|
    # This is also the swap operator if systems are ordered correctly,
    # or (vec(I))(vec(I))^dagger appropriately normalized for trace.
    # A simpler representation of J(Id_d) is sum_{k=0}^{d-1}
    # |k,k⟩⟨k,k|_{out,in} when viewing J as state.
    # As matrix: J(Id_d) has 1s at ( (i*d+j), (j*d+i) )
    # for Fortran-like ravel, or ( (i*d+j), (i*d+j) ) for C-like if map X -> X.
    # The standard Choi J(Id) = sum_{i,j} E_ij otimes E_ij.
    # For d=2, J(Id_2) = |00><00| + |01><10| + |10><01| + |11><11| (if basis |00>,|01>,|10>,|11>)
    # This is also np.eye(d*d).reshape(d,d,d,d).transpose(0,2,1,3).reshape(d*d,d*d)
    # Or more simply, for Id: L(H_d) -> L(H_d), it is the unnormalized maximally entangled state
    # projector, scaled by d. J(Id) = d * |\phi^+\rangle\langle\phi^+|
    # where |\phi^+\rangle = (1/sqrt(d)) sum_i |i,i>.
    # So J(Id) = sum_{i,j} |i,j\rangle\langle i,j|. No this is I_{d*d}.

    # Let's use the fact that J(Id) is related to the swap operator.
    # vec(Phi(X)) = (I_Y otimes K(Phi)) vec(X) where K is natural representation.
    # J(Phi)_{ij,kl} = Phi(E_lj)_ik where E_lj = |l><j|.
    # J(Id)_{ij,kl} = (Id(E_lj))_ik = (E_lj)_ik = delta_il * delta_jk.
    # This means J(Id) is the matrix for SWAP(d,d).
    # J(Id) = sum_{i,j} |i,j><j,i|

    choi_identity = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            ket_i = basis(d, i)
            ket_j = basis(d, j)
            term = np.kron(np.outer(ket_i, ket_j), np.outer(ket_j, ket_i))
            choi_identity += term

    # Alternate (simpler) construction for J(Id_d) if it's just the SWAP matrix:
    # It's the matrix that maps vec(X) to vec(X^T) after some conventions,
    # or maps |ij> to |ji>.
    # Let's be very explicit for d=2:
    # J(Id_2) matrix with basis |00>,|01>,|10>,|11|:
    # E00->E00, E01->E01, E10->E10, E11->E11
    # J(Id_2)[(0*2+0), (0*2+0)] = 1 (for E00 mapped to E00's 00 element)
    # J(Id_2)[(0*2+1), (1*2+0)] = 1 (for E10 mapped to E01's 01 element)
    # J(Id_2)[(1*2+0), (0*2+1)] = 1 (for E01 mapped to E10's 10 element)
    # J(Id_2)[(1*2+1), (1*2+1)] = 1 (for E11 mapped to E11's 11 element)
    # This is precisely np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])

    j_id = (
        np.eye(d * d).reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d * d, d * d)
    )  # This is common representation

    result = apply_channel(x_op, j_id)

    print(f"Input X:\n{x_op}")
    print(f"Choi J(Id):\n{j_id}")
    print(f"Result Phi(X):\n{result}")

    assert result.shape == x_op.shape, "Output shape mismatch"
    assert np.allclose(result, x_op), "Identity channel did not return input"
