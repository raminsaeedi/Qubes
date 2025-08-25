# src/calculation.py

import itertools
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from numba import njit
import streamlit as st

@st.cache_data
def perform_gauss_quadrature(gauss_n, cube_size):
    """
    Performs the 6D Gaussian Quadrature and Prideaux recursion.
    The cube_size determines the dimensions of the two unit cubes.
    """
    # The size 'd' of the 8 sub-cubes is half the main cube size
    d = cube_size / 2.0
    
    nodes, w = leggauss(gauss_n)
    nodes_transformed = 0.5 * (nodes + 1.0) * d
    weights_transformed = 0.5 * w * d
    
    cache = {}

    @njit(fastmath=True)
    def gauss_6d(offset, nodes_func, weights_func):
        n = nodes_func.size
        res = 0.0
        for i in range(n):
            xi, wx = nodes_func[i], weights_func[i]
            for j in range(n):
                yj, wy = nodes_func[j], weights_func[j]
                for k in range(n):
                    zk, wz = nodes_func[k], weights_func[k]
                    w1 = wx * wy * wz
                    for p in range(n):
                        x2, wxp = offset[0] + nodes_func[p], weights_func[p]
                        dx = x2 - xi
                        for q in range(n):
                            y2, wyq = offset[1] + nodes_func[q], weights_func[q]
                            dy = y2 - yj
                            for r in range(n):
                                z2, wzr = offset[2] + nodes_func[r], weights_func[r]
                                dz = z2 - zk
                                r2 = dx*dx + dy*dy + dz*dz
                                if r2 > 1e-12:
                                    res += w1 * wxp * wyq * wzr * dx / (r2**1.5)
        return res

    def pair_force(offset):
        key = tuple(round(v, 10) for v in offset)
        if key not in cache:
            cache[key] = gauss_6d(np.array(offset), nodes_transformed, weights_transformed)
        return cache[key]

    def classify(dx, dy, dz):
        ax, ay, az = abs(dx), abs(dy), abs(dz)
        if math.isclose(ax, d) and math.isclose(ay, 0) and math.isclose(az, 0): return 'F'
        if ((math.isclose(ax,d) and math.isclose(ay,d) and math.isclose(az,0)) or
            (math.isclose(ax,d) and math.isclose(az,d) and math.isclose(ay,0)) or
            (math.isclose(ay,d) and math.isclose(az,d) and math.isclose(ax,0))): return 'E'
        if all(math.isclose(a,d) for a in (ax,ay,az)): return 'V'
        return 'S'

    def S_sum(basis):
        S = 0.0
        for i1, i2, i3, j1, j2, j3 in itertools.product((0, 1), repeat=6):
            off = (basis[0] + (j1 - i1) * d,
                   basis[1] + (j2 - i2) * d,
                   basis[2] + (j3 - i3) * d)
            if classify(*off) == 'S':
                S += pair_force(off)
        return S

    # The basis vectors for the sums now depend on the cube_size
    S_F = S_sum((cube_size, 0.0, 0.0))
    S_E = S_sum((cube_size, cube_size, 0.0))
    S_V = S_sum((cube_size, cube_size, cube_size))
    
    # Prideaux-Rekursion
    V = (16/15) * S_V
    E = (V + 8*S_E) / 7
    F = (2*E + V + 4*S_F) / 3
    
    return F, S_F, S_E, S_V, V, E, len(cache)