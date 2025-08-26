# src/visualization.py

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_cube(ax, origin, size, color='blue', alpha=0.7):
    """Zeichnet einen einzelnen Würfel in 3D."""
    if isinstance(size, (int, float)):
        size = np.array([size, size, size])
        
    x, y, z = origin
    vertices = [
        [x, y, z], [x+size[0], y, z], [x+size[0], y+size[1], z], [x, y+size[1], z],
        [x, y, z+size[2]], [x+size[0], y, z+size[2]], [x+size[0], y+size[1], z+size[2]], [x, y+size[1], z+size[2]]
    ]
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]], [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]], [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]], [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]
    poly3d = [[face for face in faces]]
    ax.add_collection3d(Poly3DCollection(poly3d[0], facecolors=color, alpha=alpha, edgecolors='black', linewidths=0.8))

def plot_simulation_scene(size1, size2, gap):
    """Visualisiert die Szene mit zwei Würfeln, unterschiedlichen Größen und Abstand."""
    fig = plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-darkgrid')
    ax = fig.add_subplot(111, projection='3d')
    
    draw_cube(ax, (0, 0, 0), size1, color='lightblue')
    draw_cube(ax, (size1 + gap, 0, 0), size2, color='lightcoral')

    center1 = size1 / 2
    center2 = size1 + gap + (size2 / 2)
    arrow_length = max(size1, size2) * 0.4
    ax.quiver(center1, center1, center1, arrow_length, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2)
    ax.quiver(center2, center2, center2, -arrow_length, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2)
    
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Simulations-Szene')
    
    max_dim = max(size1 + gap + size2, size1, size2) * 1.5
    ax.set_xlim(-max_dim*0.1, max_dim); ax.set_ylim(-max_dim*0.1, max_dim); ax.set_zlim(-max_dim*0.1, max_dim)
    ax.view_init(elev=20, azim=30)
    ax.set_aspect('auto')
    plt.tight_layout()
    return fig

def plot_prideaux_method(cube_size):
    """Visualisiert die Prideaux-Methode der Zerlegung."""
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
        
    ax1.set_title('1. Zerlegung in 8x8 Teilwürfel')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=30)
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(122)
    pair_types = {'Face': 4, 'Edge': 8, 'Vertex': 4, 'Separated': 48}
    ax2.bar(pair_types.keys(), pair_types.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax2.set_title('2. Klassifizierung der 64 Paare')
    ax2.set_ylabel('Anzahl der Paare')
    plt.tight_layout()
    return fig