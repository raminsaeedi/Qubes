# app.py

import streamlit as st
import time

# Import the functions from our new modules
from src.calculation import perform_gauss_quadrature
from src.visualization import plot_original_problem, plot_prideaux_method

# Configure the Streamlit page
st.set_page_config(layout="wide", page_title="Gravitation zwischen Würfeln", page_icon="🧊")

st.title("🧊 Interaktive Simulation der Gravitationskraft zwischen Würfeln")
st.markdown("Eine visuelle Erkundung des **Trefethen-Problems Nr. 5**, gelöst mit der **Prideaux-Methode**.")

# --- Sidebar for user settings ---
with st.sidebar:
    st.header("⚙️ Simulations-Parameter")
    
    # NEW: Interactive input for cube size
    cube_size = st.number_input(
        "Kantenlänge der Würfel", 
        min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        help="Definiert die Kantenlänge der beiden Hauptwürfel."
    )
    
    gauss_n = st.slider(
        "Gauß-Quadratur Ordnung (N)", 
        min_value=2, max_value=8, value=6,
        help="Anzahl der Stützpunkte pro Dimension. Höhere Werte sind genauer, aber **deutlich** langsamer."
    )
    
    total_points = gauss_n**6
    st.info(f"Aktuelle Konfiguration:\n- **Kantenlänge:** {cube_size}\n- **Ordnung N:** {gauss_n}\n- **Gesamtpunkte:** {total_points:,}")
    
    if gauss_n > 6:
        st.warning("⚠️ Achtung: Ordnungen über 6 können sehr lange dauern!")

# --- Main tabs for the explanation ---
tab1, tab2, tab3 = st.tabs([
    "1. 🤔 Das Problem", 
    "2. 💡 Die Methode", 
    "3. ✅ Das Ergebnis"
])

with tab1:
    st.header("Das physikalische Problem: Anziehungskraft zweier Würfel")
    st.pyplot(plot_original_problem(cube_size))
    st.error(
        "**Die Herausforderung:** Das Integral zur Berechnung der Kraft hat eine **Singularität**. "
        "An der Berührungsfläche ist der Abstand zwischen den Punkten null."
    )

with tab2:
    st.header("Die Prideaux-Methode: Ein genialer Trick")
    st.markdown(
        "Anstatt das singuläre Problem direkt zu lösen, zerlegen wir jeden Würfel in 8 kleinere Teilwürfel "
        "und nutzen eine rekursive Beziehung, um die Singularität zu umgehen."
    )
    st.pyplot(plot_prideaux_method(cube_size))

with tab3:
    st.header("Das Ergebnis der Simulation")
    
    # Perform the calculation with the user-defined parameters
    with st.spinner(f"Berechne Kraft für Würfel der Größe {cube_size} mit {total_points:,} Stützpunkten..."):
        start_time = time.time()
        F, S_F, S_E, S_V, V, E, cache_len = perform_gauss_quadrature(gauss_n, cube_size)
        duration = time.time() - start_time

    st.success(f"Berechnung in {duration:.2f} Sekunden abgeschlossen!")
    
    st.subheader("🎉 Finale Gravitationskraft")
    # The result 'F' is dimensionless here. For unit cubes, the literature value is ~0.926.
    # The force scales with the 4th power of the cube size (F ∝ L⁴). We normalize it for comparison.
    normalized_F = F / (cube_size**4)
    
    col1, col2 = st.columns(2)
    col1.metric(
        label=f"Berechnete Kraft F für Kantenlänge {cube_size}",
        value=f"{F:.8f}"
    )
    col2.metric(
        label=f"Normalisierte Kraft (F/L⁴)",
        value=f"{normalized_F:.8f}",
        help="Die Kraft skaliert mit der 4. Potenz der Kantenlänge. Dieser Wert ist zum Vergleich mit dem bekannten Ergebnis für Einheitswürfel (~0.926)."
    )

    with st.expander("Details der Berechnung anzeigen"):
        st.markdown(f"**S-Summen (Beiträge der getrennten Paare):**")
        st.code(f"S_F (Face): {S_F:.8f}\nS_E (Edge): {S_E:.8f}\nS_V (Vertex): {S_V:.8f}")
        
        st.markdown(f"**Rekursions-Ergebnisse:**")
        st.code(f"V (Vertex-Kraft): {V:.8f}\nE (Edge-Kraft):   {E:.8f}")
        
        st.info(f"Für die Berechnung wurden {cache_len} einzigartige Abstände zwischengespeichert.")