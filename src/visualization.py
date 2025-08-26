# src/visualization.py

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_cube(ax, origin, size, color='blue', alpha=0.7):
    if isinstance(size, (int, float)): size = np.array([size, size, size])
    x, y, z = origin
    vertices = [[x,y,z],[x+size[0],y,z],[x+size[0],y+size[1],z],[x,y+size[1],z],[x,y,z+size[2]],[x+size[0],y,z+size[2]],[x+size[0],y+size[1],z+size[2]],[x,y+size[1],z+size[2]]]
    faces = [[vertices[0],vertices[1],vertices[2],vertices[3]],[vertices[4],vertices[5],vertices[6],vertices[7]],[vertices[0],vertices[1],vertices[5],vertices[4]],[vertices[2],vertices[3],vertices[7],vertices[6]],[vertices[1],vertices[2],vertices[6],vertices[5]],[vertices[4],vertices[7],vertices[3],vertices[0]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, alpha=alpha, edgecolors='black', linewidths=0.8))

def plot_simulation_scene(size1, size2, gap):
    fig = plt.figure(figsize=(10, 6)); ax = fig.add_subplot(111, projection='3d')
    draw_cube(ax, (0, 0, 0), size1, color='lightblue')
    draw_cube(ax, (size1 + gap, 0, 0), size2, color='lightcoral')
    center1, center2 = size1 / 2, size1 + gap + (size2 / 2)
    arrow_len = max(size1, size2) * 0.4
    ax.quiver(center1, center1, center1, arrow_len, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2)
    ax.quiver(center2, center2, center2, -arrow_len, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2)
    max_dim = max(size1 + gap + size2, size1, size2) * 1.5
    ax.set_title('Simulations-Szene'); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(-max_dim*0.1, max_dim); ax.set_ylim(-max_dim*0.1, max_dim); ax.set_zlim(-max_dim*0.1, max_dim)
    ax.view_init(elev=20, azim=30); ax.set_aspect('auto'); plt.tight_layout()
    return fig

def plot_prideaux_flow(S_F, S_E, S_V, V, E, F):
    fig, ax = plt.subplots(figsize=(8, 10)); ax.axis('off')
    boxes = [{'xy':(0.5,0.9),'text':f'S_V = {S_V:.6f}','color':'lightgreen'}, {'xy':(0.5,0.75),'text':f'V=(16/15)S_V={V:.6f}','color':'lightblue'},
             {'xy':(0.5,0.6),'text':f'S_E = {S_E:.6f}','color':'lightgreen'}, {'xy':(0.5,0.45),'text':f'E=(V+8S_E)/7={E:.6f}','color':'lightblue'},
             {'xy':(0.5,0.3),'text':f'S_F = {S_F:.6f}','color':'lightgreen'}, {'xy':(0.5,0.15),'text':f'F=(2E+V+4S_F)/3={F:.6f}','color':'lightcoral'}]
    for i, box in enumerate(boxes):
        ax.text(box['xy'][0], box['xy'][1], box['text'], ha='center', va='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor=box['color'], edgecolor='black', linewidth=1))
        if i < len(boxes) - 1: ax.annotate('', xy=(box['xy'][0], box['xy'][1] - 0.05), xytext=(boxes[i+1]['xy'][0], boxes[i+1]['xy'][1] + 0.05), arrowprops=dict(arrowstyle='->', lw=2.5, color='gray'))
    ax.set_xlim(0, 1); ax.set_ylim(0.05, 1); ax.set_title('Berechnungsfluss der Rekursion', fontsize=14, fontweight='bold'); plt.tight_layout()
    return fig

def plot_prideaux_method_decomposition(cube_size):
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    sub_size = cube_size / 2.0
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, 8)); colors2 = plt.cm.Reds(np.linspace(0.4, 0.9, 8))
    for i, (x,y,z) in enumerate(itertools.product([0,sub_size],repeat=3)): draw_cube(ax, (x,y,z), sub_size, color=colors1[i])
    for i, (x,y,z) in enumerate(itertools.product([cube_size,cube_size+sub_size],repeat=3)): draw_cube(ax, (x,y,z), sub_size, color=colors2[i])
    ax.set_title('Zerlegung in 8x8 Teilwürfel'); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=30); ax.set_aspect('equal'); plt.tight_layout()
    return fig

def plot_method_comparison():
    """
    Creates a side-by-side plot comparing the concepts of Direct Integration and the Prideaux Method.
    """
    fig = plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-darkgrid')

    # --- Left Plot: Direct Integration Concept ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Methode 1: Direkte Integration\n(Für Würfel mit Abstand)", pad=20)

    # Draw two cubes with a gap
    size1, size2, gap = 0.8, 1.0, 0.4
    draw_cube(ax1, (0, 0, 0), size1, color='lightblue', alpha=0.5)
    draw_cube(ax1, (size1 + gap, 0, 0), size2, color='lightcoral', alpha=0.5)

    # Show random sampling points
    np.random.seed(42)
    points1 = np.random.rand(20, 3) * size1
    points2 = np.random.rand(20, 3) * size2 + np.array([size1 + gap, 0, 0])
    ax1.scatter(points1[:,0], points1[:,1], points1[:,2], c='blue', s=10)
    ax1.scatter(points2[:,0], points2[:,1], points2[:,2], c='red', s=10)

    # Draw a line between two sample points
    ax1.plot([points1[0,0], points2[0,0]], [points1[0,1], points2[0,1]], [points1[0,2], points2[0,2]], 'k--')
    ax1.text(1.0, 1.0, 1.0, "Summe aller Kräfte\nzwischen allen Punkten", fontsize=9, ha='center')
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
    ax1.view_init(elev=20, azim=30)


    # --- Right Plot: Prideaux Method Concept ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Methode 2: Prideaux-Verfahren\n(Für berührende, identische Würfel)", pad=20)
    
    # Draw the sub-cubes
    cube_size = 1.0
    sub_size = cube_size / 2.0
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
    for i, (x, y, z) in enumerate(itertools.product([0, sub_size], repeat=3)):
        draw_cube(ax2, (x, y, z), sub_size, color=colors1[i], alpha=0.8)
        
    colors2 = plt.cm.Reds(np.linspace(0.4, 0.9, 8))
    for i, (x, y, z) in enumerate(itertools.product([cube_size, cube_size + sub_size], repeat=3)):
        draw_cube(ax2, (x, y, z), sub_size, color=colors2[i], alpha=0.8)

    ax2.text(1.0, 1.3, 1.3, "1. Zerlegen & nur Kräfte\n   getrennter Paare berechnen\n2. Gesamtkraft rekonstruieren", fontsize=9, ha='center')
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    ax2.view_init(elev=20, azim=30)
    
    plt.tight_layout()
    return fig
