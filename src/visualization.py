# src/visualization.py

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_cube(ax, origin, size, color='blue', alpha=0.7):
    """Zeichnet einen einzelnen Würfel in 3D."""
    x, y, z = origin
    vertices = [
        [x, y, z], [x+size, y, z], [x+size, y+size, z], [x, y+size, z],
        [x, y, z+size], [x+size, y, z+size], [x+size, y+size, z+size], [x, y+size, z+size]
    ]
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]], [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]], [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]], [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]
    poly3d = [[face for face in faces]]
    ax.add_collection3d(Poly3DCollection(poly3d[0], facecolors=color, alpha=alpha, edgecolors='black', linewidths=0.8))

def plot_original_problem(cube_size):
    fig = plt.figure(figsize=(12, 5))
    plt.style.use('seaborn-v0_8-darkgrid')
    ax1 = fig.add_subplot(111, projection='3d')
    
    draw_cube(ax1, (0, 0, 0), cube_size, color='lightblue')
    draw_cube(ax1, (cube_size, 0, 0), cube_size, color='lightcoral')

    center1 = cube_size / 2
    center2 = cube_size * 1.5
    arrow_length = cube_size * 0.4
    ax1.quiver(center1, center1, center1, arrow_length, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2)
    ax1.quiver(center2, center1, center1, -arrow_length, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2)
    
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('Physikalisches Problem')
    
    lim_min, lim_max = -0.5 * cube_size, 2.5 * cube_size
    ax1.set_xlim(lim_min, lim_max); ax1.set_ylim(lim_min, lim_max); ax1.set_zlim(lim_min, lim_max)
    ax1.view_init(elev=20, azim=30)
    ax1.set_aspect('equal')
    plt.tight_layout()
    return fig

def plot_prideaux_method(cube_size):
    fig = plt.figure(figsize=(12, 5))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    ax1 = fig.add_subplot(121, projection='3d')
    sub_size = cube_size / 2.0
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
    for i, (x, y, z) in enumerate(itertools.product([0, sub_size], repeat=3)):
        draw_cube(ax1, (x, y, z), sub_size, color=colors1[i])
        
    colors2 = plt.cm.Reds(np.linspace(0.4, 0.9, 8))
    for i, (x, y, z) in enumerate(itertools.product([cube_size, cube_size + sub_size], repeat=3)):
        draw_cube(ax1, (x, y, z), sub_size, color=colors2[i])
        
    ax1.set_title('1. Zerlegung in 64 Teil-Würfelpaare')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=30)
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(122)
    pair_types = {'Face (F)': 4, 'Edge (E)': 8, 'Vertex (V)': 4, 'Separated (S)': 48}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax2.bar(pair_types.keys(), pair_types.values(), color=colors, alpha=0.8)
    ax2.set_title('2. Klassifizierung der Paare')
    ax2.set_ylabel('Anzahl der Paare')
    for i, (k, v) in enumerate(pair_types.items()):
        ax2.text(i, v + 1, str(v), ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig