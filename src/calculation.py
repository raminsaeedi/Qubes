# src/calculation.py

import itertools
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from numba import njit
import streamlit as st

# ==============================================================================
# METHODE 1: PRIDEAUX-REKURSION (Für identische, berührende Würfel)
# ==============================================================================

@st.cache_data
def calculate_force_prideaux(gauss_n, cube_size):
    """
    Berechnet die Kraft mit der hochpräzisen Prideaux-Methode.
    NUR für identische, berührende Würfel (L1=L2, Abstand=0).
    """
    d = cube_size / 2.0
    
    nodes, w = leggauss(gauss_n)
    nodes_transformed = 0.5 * (nodes + 1.0) * d
    weights_transformed = 0.5 * w * d
    
    cache = {}

    @njit(fastmath=True)
    def gauss_6d_prideaux(offset, nodes_func, weights_func):
        # (Numba-Code bleibt derselbe wie zuvor)
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
            cache[key] = gauss_6d_prideaux(np.array(offset), nodes_transformed, weights_transformed)
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
            off = (basis[0] + (j1 - i1) * d, basis[1] + (j2 - i2) * d, basis[2] + (j3 - i3) * d)
            if classify(*off) == 'S':
                S += pair_force(off)
        return S

    S_F = S_sum((cube_size, 0.0, 0.0))
    S_E = S_sum((cube_size, cube_size, 0.0))
    S_V = S_sum((cube_size, cube_size, cube_size))
    
    V = (16/15) * S_V
    E = (V + 8*S_E) / 7
    F = (2*E + V + 4*S_F) / 3
    
    return F


# ==============================================================================
# METHODE 2: DIREKTE INTEGRATION (Für beliebige Größen und Abstände)
# ==============================================================================

@st.cache_data
def calculate_force_direct(gauss_n, size1, size2, gap):
    """
    Berechnet die Kraft durch direkte 6D-Integration.
    Funktioniert für beliebige Größen und Abstände.
    """
    nodes, w = leggauss(gauss_n)

    # Skaliere Knoten und Gewichte für jeden Würfel separat
    nodes1 = 0.5 * (nodes + 1.0) * size1
    weights1 = 0.5 * w * size1
    
    # Würfel 2 ist verschoben
    nodes2_x = 0.5 * (nodes + 1.0) * size2 + size1 + gap
    nodes2_yz = 0.5 * (nodes + 1.0) * size2
    weights2 = 0.5 * w * size2
    
    @njit(fastmath=True)
    def gauss_6d_direct(nodes1, weights1, nodes2_x, nodes2_yz, weights2):
        n = nodes1.size
        res = 0.0
        for i in range(n):
            x1, w1x = nodes1[i], weights1[i]
            for j in range(n):
                y1, w1y = nodes1[j], weights1[j]
                for k in range(n):
                    z1, w1z = nodes1[k], weights1[k]
                    w_total1 = w1x * w1y * w1z
                    for p in range(n):
                        x2, w2x = nodes2_x[p], weights2[p]
                        dx = x2 - x1
                        for q in range(n):
                            y2, w2y = nodes2_yz[q], weights2[q]
                            dy = y2 - y1
                            for r in range(n):
                                z2, w2z = nodes2_yz[r], weights2[r]
                                dz = z2 - z1
                                w_total2 = w2x * w2y * w2z
                                r2 = dx*dx + dy*dy + dz*dz
                                if r2 > 1e-12:
                                    res += w_total1 * w_total2 * dx / (r2**1.5)
        return res

    force = gauss_6d_direct(nodes1, weights1, nodes2_x, nodes2_yz, weights2)
    return force