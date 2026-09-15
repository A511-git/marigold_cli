import numpy as np

__all__ = ['create_icosahedron_mesh']


def create_icosahedron_mesh():
    """
    Create an icosahedron mesh centered at origin.
    """
    A = (1 + 5 ** 0.5) / 2
    vertices = np.array([
        [0, 1, A], [0, -1, A], [0, 1, -A], [0, -1, -A],
        [1, A, 0], [-1, A, 0], [1, -A, 0], [-1, -A, 0],
        [A, 0, 1], [A, 0, -1], [-A, 0, 1], [-A, 0, -1]
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 8], [0, 8, 4], [0, 4, 5], [0, 5, 10], [0, 10, 1],
        [3, 2, 9], [3, 9, 6], [3, 6, 7], [3, 7, 11], [3, 11, 2],
        [1, 6, 8], [8, 9, 4], [4, 2, 5], [5, 11, 10], [10, 7, 1],
        [2, 4, 9], [9, 8, 6], [6, 1, 7], [7, 10, 11], [11, 5, 2]
    ], dtype=np.int32)
    return vertices, faces