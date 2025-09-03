import scipy
from scipy.sparse import lil_matrix
import numpy as np

def unrotated_surface_code(d: int):
    assert d > 2, "Distance of the code must be greaater than 2"

    # For unrotated surface code with distance 'd' the number of physical qubits is 2d(d- 1)
    num_qubits = 2 * d * ( d - 1 )

    # Horizontal edges: (row i in [0..d-1], col j in [0..d-2]) between (i,j) -- (i,j+1)
    hor_id = {}
    vid = 0
    for i in range(d):
        for j in range(d-1):
            hor_id[(i,j)] = vid
            vid += 1

    # Vertical edges: (row i in [0..d-2], col j in [0..d-1]) between (i,j) -- (i+1,j)
    ver_id = {}
    for i in range(d-1):
        for j in range(d):
            ver_id[(i,j)] = vid
            vid += 1

    if not num_qubits == vid:
        raise ValueError("Number of Qubits Mismatch")
    
    # HZ matrix
    # For HZ matrix the number of rows is equal to the number of faces
    num_faces = (d - 1)*(d - 1)
    HZ = lil_matrix((num_faces, num_qubits), dtype=int)

    row = 0 


    for i in range(d - 1):
        for j in range(d - 1):
            HZ[row, hor_id[(i, j)]] = 1 #top
            HZ[row, ver_id[(i, j+1)]] = 1 #right
            HZ[row, hor_id[(i+1, j)]] = 1 #bottom
            HZ[row, ver_id[(i, j)]] = 1 #left

            row += 1

        HZ = HZ.tocsr()

    # For HX matrix the number of rows is equal to the number of vertices
    num_vertices = d*d
    HX = lil_matrix((num_vertices, num_qubits), dtype=int)

    row = 0
    for i in range(d):
        for j in range(d):
            # incident horizontal edges
            if j > 0:
                HX[row, hor_id[(i, j-1)]] = 1   # left
            if j < d-1:
                HX[row, hor_id[(i, j)]] = 1     # right
            # incident vertical edges
            if i > 0:
                HX[row, ver_id[(i-1, j)]] = 1    # up
            if i < d-1:
                HX[row, ver_id[(i, j)]] = 1      # down
            row += 1

    HX = HX.tocsr()

    # Checking CSS condition
    if np.any((HX @ HZ.T).toarray() % 2 != 0):
        raise ValueError("CSS condition violated")

    return HZ, HX, num_faces, num_qubits, num_vertices

HZ, HX, num_faces, num_qubits, num_vertices = unrotated_surface_code(4)

print(f'Number of faces: {num_faces}')
print(f'Number of qubits: {num_qubits}')
print(f'Number of vertices: {num_vertices}')
print("HZ:",HZ.toarray())
print("HX:",HX.toarray())
