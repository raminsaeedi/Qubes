import streamlit as st
import itertools, math
import numpy as np
from numpy.polynomial.legendre import leggauss
from numba import njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyBboxPatch
import time
import pandas as pd

# =================================================================================
# KERNBERECHNUNGS-FUNKTIONEN (Kombiniert aus deinen Skripten)
# =================================================================================

# Konfiguration der Streamlit-Seite
st.set_page_config(layout="wide", page_title="Gravitation zwischen Würfeln", page_icon="🧊")

# Caching für die rechenintensive Gauss-Quadratur
# @st.cache_data wird die Funktion nur einmal für dieselben Eingabeparameter ausführen.
@st.cache_data
def perform_gauss_quadrature(gauss_n, d):
    """Führt die 6D-Gauß-Quadratur durch, um die S-Werte zu berechnen."""
    nodes, w = leggauss(gauss_n)
    nodes_transformed = 0.5 * (nodes + 1.0) * d
    weights_transformed = 0.5 * w * d
    
    # Memo-Cache, damit jedes Offset nur 1× integriert wird
    # Dieser Cache ist lokal zur Funktion und wird bei jedem Lauf zurückgesetzt
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
                                if r2 > 1e-12:  # Singularität vermeiden
                                    res += w1 * wxp * wyq * wzr * dx / (r2**1.5)
        return res

    def pair_force(offset):
        key = tuple(round(v, 10) for v in offset)
        if key not in cache:
            cache[key] = gauss_6d(np.array(offset), nodes_transformed, weights_transformed)
        return cache[key]

    def classify(dx, dy, dz):
        ax, ay, az = abs(dx), abs(dy), abs(dz)
        if math.isclose(ax, d) and ay == az == 0.0: return 'F'
        if ((math.isclose(ax,d) and math.isclose(ay,d) and az==0.0) or
            (math.isclose(ax,d) and math.isclose(az,d) and ay==0.0) or
            (math.isclose(ay,d) and math.isclose(az,d) and ax==0.0)): return 'E'
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

    # Berechnung der S-Werte
    S_F = S_sum((1.0, 0.0, 0.0))
    S_E = S_sum((1.0, 1.0, 0.0))
    S_V = S_sum((1.0, 1.0, 1.0))
    
    return S_F, S_E, S_V, len(cache)

# =================================================================================
# VISUALISIERUNGS-FUNKTIONEN
# =================================================================================

def draw_cube(ax, origin, size, color='blue', alpha=0.7, label=None):
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
    ax.add_collection3d(Poly3DCollection(poly3d[0], facecolors=color, alpha=alpha, edgecolors='black', linewidths=1))
    if label:
        ax.text(x + size/2, y + size/2, z + size/2, label, fontsize=12, ha='center', va='center')

def plot_original_problem():
    """Visualisiert das ursprüngliche Problem mit zwei Würfeln und der Formel."""
    fig = plt.figure(figsize=(12, 4))
    plt.style.use('seaborn-v0_8-darkgrid')

    # 3D Darstellung
    ax1 = fig.add_subplot(121, projection='3d')
    draw_cube(ax1, (0, 0, 0), 1.0, color='lightblue', label='Würfel 1')
    draw_cube(ax1, (1, 0, 0), 1.0, color='lightcoral', label='Würfel 2')
    ax1.quiver(0.5, 0.5, 0.5, 0.4, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2)
    ax1.quiver(1.5, 0.5, 0.5, -0.4, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('Physikalisches Problem')
    ax1.set_xlim(-0.5, 2.5); ax1.set_ylim(-0.5, 1.5); ax1.set_zlim(-0.5, 1.5)
    ax1.view_init(elev=20, azim=30)
    
    # Formel
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    formula = r'$F = \int_{W1} \int_{W2} \frac{(x_2 - x_1) \, dV_1 dV_2}{| \vec{r}_1 - \vec{r}_2 |^3}$'
    ax2.text(0.5, 0.6, "Die Herausforderung:", fontsize=14, ha='center')
    ax2.text(0.5, 0.4, formula, fontsize=18, ha='center')
    ax2.text(0.5, 0.2, r'Das Integral ist singulär, da sich die Würfel berühren ($|\vec{r}_1 - \vec{r}_2| \to 0$).',
             fontsize=10, ha='center', wrap=True)
    plt.tight_layout()
    return fig

def plot_prideaux_method():
    """Visualisiert die Prideaux-Methode der Zerlegung."""
    fig = plt.figure(figsize=(12, 5))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Zerlegte Würfel
    ax1 = fig.add_subplot(121, projection='3d')
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
    for i, (x, y, z) in enumerate(itertools.product([0, 0.5], repeat=3)):
        draw_cube(ax1, (x, y, z), 0.5, color=colors1[i])
    colors2 = plt.cm.Reds(np.linspace(0.4, 0.9, 8))
    for i, (x, y, z) in enumerate(itertools.product([1, 1.5], repeat=3)):
        draw_cube(ax1, (x, y, z), 0.5, color=colors2[i])
    ax1.set_title('1. Zerlegung in 64 Teil-Würfelpaare')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.view_init(elev=20, azim=30)
    
    # Paartypen
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

def plot_gauss_quadrature(gauss_n, d):
    """Visualisiert die Gauß-Knoten und Gewichte."""
    nodes, w = leggauss(gauss_n)
    nodes_transformed = 0.5 * (nodes + 1.0) * d
    weights_transformed = 0.5 * w * d
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1D Knoten
    ax1.plot([0, d], [0.5, 0.5], 'k--', alpha=0.3)
    ax1.scatter(nodes_transformed, np.ones_like(nodes_transformed) * 0.5, 
                c=weights_transformed, cmap='viridis', s=150, zorder=5, edgecolors='k')
    ax1.set_title(f'1D Gauß-Knoten (Ordnung {gauss_n}) auf [0, {d}])')
    ax1.set_xlabel('Position')
    ax1.set_yticks([])
    
    # 2D Gitter
    X, Y = np.meshgrid(nodes_transformed, nodes_transformed)
    ax2.scatter(X.flatten(), Y.flatten(), s=50, c='red', alpha=0.8)
    ax2.set_title(f'2D Gauß-Gitter ({gauss_n}x{gauss_n} = {gauss_n**2} Punkte)')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def plot_final_result(F, S_F, S_E, S_V, V, E):
    """Visualisiert das Endergebnis und die Komponenten."""
    fig = plt.figure(figsize=(12, 5))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Komponenten
    ax1 = fig.add_subplot(121)
    components = ['S_V', 'S_E', 'S_F', 'V', 'E', 'F']
    values = [S_V, S_E, S_F, V, E, F]
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    bars = ax1.bar(components, values, color=colors, alpha=0.9)
    ax1.set_title('Komponenten der Prideaux-Rekursion')
    ax1.set_ylabel('Wert')
    ax1.bar_label(bars, fmt='{:.4f}', padding=3)

    # Vergleich
    ax2 = fig.add_subplot(122)
    trefethen_result = 0.9259812606
    methods = ['Unsere Berechnung', 'Literatur (Trefethen)']
    results = [F, trefethen_result]
    colors = ['#2ca02c', '#d62728']
    bars = ax2.bar(methods, results, color=colors, alpha=0.9, width=0.5)
    ax2.set_title('Vergleich mit dem exakten Wert')
    ax2.set_ylabel('Gravitationskraft F')
    ax2.set_ylim(min(results)*0.9999, max(results)*1.0001)
    ax2.bar_label(bars, fmt='{:.10f}', padding=3)
    
    plt.tight_layout()
    return fig

# =================================================================================
# STREAMLIT APP LAYOUT
# =================================================================================

st.title("🧊 Interaktive Simulation der Gravitationskraft zwischen Würfeln")
st.markdown("Eine visuelle Erkundung des **Trefethen-Problems Nr. 5**, gelöst mit der **Prideaux-Methode** und **Gauß-Quadratur**.")

# --- Sidebar für Einstellungen ---
with st.sidebar:
    st.header("⚙️ Simulations-Parameter")
    st.markdown("Passe die Genauigkeit der numerischen Integration an.")
    
    gauss_n = st.slider(
        "Gauß-Quadratur Ordnung (N)", 
        min_value=2, max_value=8, value=6,
        help="Anzahl der Stützpunkte pro Dimension. Höhere Werte sind genauer, aber **deutlich** langsamer. "
             f"Die Gesamtanzahl der Punkte ist N⁶. Bei N=6 sind es bereits {6**6:,} Punkte!"
    )
    
    d = 0.5 # Kantenlänge der Teilwürfel, fest für dieses Problem
    total_points = gauss_n**6
    
    st.info(f"Aktuelle Konfiguration:\n- **Ordnung N:** {gauss_n}\n- **Gesamtpunkte:** {total_points:,}")
    
    if gauss_n > 6:
        st.warning("⚠️ Achtung: Ordnungen über 6 können sehr lange Rechenzeiten verursachen (Minuten!).")

# --- Haupt-Tabs für die Erklärung ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. 🤔 Das Problem", 
    "2. 💡 Die Methode", 
    "3. 💻 Die Berechnung", 
    "4. ✅ Das Ergebnis",
    "5. 📜 Der Code"
])

with tab1:
    st.header("Das physikalische Problem: Anziehungskraft zweier Würfel")
    st.markdown(
        "Wir wollen die Gravitationskraft zwischen zwei identischen, sich berührenden Einheitswürfeln berechnen. "
        "Stell dir vor, zwei perfekt würfelförmige Asteroiden berühren sich im All. Wie stark ziehen sie sich an?"
    )
    st.pyplot(plot_original_problem())
    st.error(
        "**Die große Herausforderung:** Das Integral zur Berechnung der Kraft hat eine **Singularität**. "
        "An der Berührungsfläche ist der Abstand zwischen den Punkten null, was zu einer Division durch Null führt. "
        "Standard-Integrationsverfahren versagen hier."
    )

with tab2:
    st.header("Die Prideaux-Methode: Ein genialer Trick")
    st.markdown(
        "Anstatt das singuläre Problem direkt zu lösen, nutzen wir einen Trick von J. Prideaux (2002):"
        "\n1. **Zerlegen:** Wir teilen jeden großen Würfel in 8 kleinere Teilwürfel auf (insgesamt 16 Teilwürfel)."
        "\n2. **Klassifizieren:** Wir betrachten die Kraft zwischen allen 64 Paaren dieser Teilwürfel. Einige berühren sich (Problem!), die meisten aber nicht (einfach!)."
        "\n3. **Rekursion:** Wir finden eine clevere rekursive Beziehung, die uns erlaubt, die Kräfte der sich berührenden Paare aus den Kräften der getrennten Paare abzuleiten. So umgehen wir die Singularität komplett!"
    )
    st.pyplot(plot_prideaux_method())

with tab3:
    st.header("Die numerische Integration: Gauß-Quadratur")
    st.markdown(
        "Um die Kraft zwischen den vielen *getrennten* Teilwürfel-Paaren zu berechnen, verwenden wir ein leistungsstarkes numerisches Verfahren: die **Gauß-Quadratur**. "
        "Sie approximiert das 6-dimensionale Integral, indem sie die Funktion an speziell gewählten 'Gauß-Knoten' auswertet."
        "\n\nUnten siehst du die Verteilung dieser Knoten für die in der Seitenleiste gewählte Ordnung."
    )
    st.pyplot(plot_gauss_quadrature(gauss_n, d))
    
    st.subheader("Berechnung der S-Werte")
    st.markdown("Jetzt führen wir die rechenintensive Simulation durch. Wir berechnen die Gesamtkraft für alle **48 getrennten ('Separated')** Paare für die drei Basiskonfigurationen (Face, Edge, Vertex).")

    # Berechnung durchführen und anzeigen
    with st.spinner(f"Berechne S-Werte mit {total_points:,} Stützpunkten... (kann einen Moment dauern)"):
        start_time = time.time()
        S_F, S_E, S_V, cache_len = perform_gauss_quadrature(gauss_n, d)
        end_time = time.time()
        duration = end_time - start_time

    st.success(f"Berechnung in {duration:.2f} Sekunden abgeschlossen!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Summe Face-Konfiguration (S_F)", f"{S_F:.8f}")
    col2.metric("Summe Edge-Konfiguration (S_E)", f"{S_E:.8f}")
    col3.metric("Summe Vertex-Konfiguration (S_V)", f"{S_V:.8f}")
    st.info(f"Für die Berechnung wurden {cache_len} einzigartige Würfelpaar-Abstände zwischengespeichert (Memoization).")

with tab4:
    st.header("Das Endergebnis: Die Rekursion auflösen")
    st.markdown(
        "Mit den berechneten S-Werten können wir nun das lineare Gleichungssystem der Prideaux-Methode lösen, um die gesuchte Kraft F (für zwei sich an der Fläche berührende Würfel) zu finden."
    )
    
    # Rekursion
    V = (16/15) * S_V
    E = (V + 8*S_E) / 7
    F = (2*E + V + 4*S_F) / 3
    
    st.latex(r"V = \frac{16}{15} S_V")
    st.latex(r"E = \frac{1}{7} (V + 8 S_E)")
    st.latex(r"F = \frac{1}{3} (2E + V + 4 S_F)")
    
    st.subheader("🎉 Finale Gravitationskraft")
    
    st.metric(
        label=f"Berechnete Kraft F (mit N={gauss_n})",
        value=f"{F:.10f}",
        delta=f"{(F - 0.9259812606) / 0.9259812606:.6e} Relativer Fehler",
        delta_color="normal"
    )
    st.pyplot(plot_final_result(F, S_F, S_E, S_V, V, E))
    
with tab5:
    st.header("Vollständiger App-Code")
    st.markdown(
        "Dieser Code kombiniert die Berechnungslogik, die Visualisierungen und das interaktive UI mit Streamlit."
    )
    with open(__file__, 'r', encoding='utf-8') as f:
        st.code(f.read(), language='python')