# app.py

import streamlit as st
import time

from src.calculation import calculate_force_prideaux, calculate_force_direct
from src.visualization import plot_simulation_scene, plot_prideaux_method

st.set_page_config(layout="wide", page_title="Gravitations-Simulator", page_icon="🧊")

st.title("🧊 Interaktiver Gravitations-Simulator für Würfel")

# --- Sidebar für Benutzereingaben ---
with st.sidebar:
    st.header("⚙️ Simulations-Parameter")
    
    st.subheader("Würfel-Geometrie")
    size1 = st.number_input("Kantenlänge Würfel 1 (L₁)", 0.1, 10.0, 1.0, 0.1)
    size2 = st.number_input("Kantenlänge Würfel 2 (L₂)", 0.1, 10.0, 1.0, 0.1)
    gap = st.number_input("Abstand", 0.0, 10.0, 0.0, 0.1, format="%.2f")
    
    st.subheader("Berechnungs-Genauigkeit")
    gauss_n = st.slider("Gauß-Quadratur Ordnung (N)", 2, 8, 6, help="Höhere Werte sind genauer, aber langsamer.")
    
    st.info(f"**Punkte pro Integral:** {gauss_n**6:,}")

# --- Logik zur Methodenauswahl ---
is_prideaux_case = (gap == 0.0 and size1 == size2)

st.header("Simulation und Ergebnis")

# Button zum Starten der Berechnung
if st.button("▶️ Gravitationskraft berechnen", type="primary"):
    
    # Führe die passende Berechnung durch
    with st.spinner(f"Berechnung läuft... (Ordnung N={gauss_n})"):
        start_time = time.time()
        if is_prideaux_case:
            force = calculate_force_prideaux(gauss_n, size1)
            method_used = "Prideaux-Methode"
        else:
            force = calculate_force_direct(gauss_n, size1, size2, gap)
            method_used = "Direkte Integration"
        duration = time.time() - start_time
    
    st.success(f"Berechnung in {duration:.2f} Sekunden abgeschlossen!")

    # --- Ergebnis-Anzeige in Spalten ---
    col1, col2 = st.columns([2, 1]) # 2/3 für die Szene, 1/3 für die Ergebnisse

    with col1:
        st.subheader("Simulations-Szene")
        # Wichtig: use_container_width=True passt die Größe an die Spalte an
        fig_scene = plot_simulation_scene(size1, size2, gap)
        st.pyplot(fig_scene, use_container_width=True)

    with col2:
        st.subheader("🎉 Ergebnis")
        st.metric(label="Berechnete Gravitationskraft F", value=f"{force:.8f}")
        
        st.subheader("Analyse")
        if is_prideaux_case:
            st.info(f"**Methode:** {method_used}\n\nEs wurde der Spezialfall für identische, berührende Würfel erkannt und die hochpräzise Prideaux-Methode verwendet.")
        else:
            st.info(f"**Methode:** {method_used}\n\nDie Kraft wurde durch direkte 6D-Integration berechnet, da die Würfel nicht identisch sind oder sich nicht berühren.")
            if gap == 0.0:
                st.warning("Da sich die Würfel berühren, kann das Ergebnis der direkten Integration durch die Singularität ungenau sein.")

    st.markdown("---") # Trennlinie

    # --- Detaillierte Visualisierungen in Tabs ---
    st.subheader("Detail-Visualisierungen")

    if is_prideaux_case:
        # Zeige die Prideaux-Visualisierung nur, wenn sie relevant ist
        tab1, tab2 = st.tabs(["Info", "Prideaux-Zerlegung"])
        with tab1:
            st.markdown("Die Prideaux-Methode zerlegt die Würfel in kleinere Einheiten, um die Singularität an der Berührungsfläche mathematisch zu umgehen. Dies führt zu einem sehr genauen Ergebnis.")
        with tab2:
            fig_prideaux = plot_prideaux_method(size1)
            st.pyplot(fig_prideaux, use_container_width=True)
    else:
        st.info("Für die direkte Integration gibt es keine weiteren Zerlegungs-Visualisierungen.")