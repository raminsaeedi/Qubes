# app.py

import streamlit as st
import time

# Import the new comparison plot function
from src.calculation import calculate_force_prideaux, calculate_force_direct
from src.visualization import plot_simulation_scene, plot_prideaux_flow, plot_prideaux_method_decomposition, plot_method_comparison

st.set_page_config(layout="wide", page_title="Gravitations-Simulator", page_icon="🧊")

st.title("🧊 Interaktiver Gravitations-Simulator für Würfel")

# --- Main Tabs for App Structure ---
tab_sim, tab_vergleich = st.tabs(["Simulation & Ergebnis", "Vergleich der Methoden"])

with tab_sim:
    # --- Sidebar for user inputs ---
    with st.sidebar:
        st.header("⚙️ Simulations-Parameter")
        size1 = st.number_input("Kantenlänge Würfel 1 (L₁)", 0.1, 10.0, 1.0, 0.1)
        size2 = st.number_input("Kantenlänge Würfel 2 (L₂)", 0.1, 10.0, 1.0, 0.1)
        gap = st.number_input("Abstand", 0.0, 10.0, 0.0, 0.1, format="%.2f")
        gauss_n = st.slider("Gauß-Quadratur Ordnung (N)", 2, 12, 8, help="Für 10-stellige Genauigkeit sind hohe Werte (N > 10) erforderlich.")
        st.info(f"**Punkte pro Integral:** {gauss_n**6:,}")
        st.warning("**Achtung:** Werte für N > 8 können **sehr lange** Rechenzeiten haben (mehrere Minuten!).")

    # --- Logic to select the calculation method ---
    is_prideaux_case = (gap == 0.0 and size1 == size2)

    st.header("Simulation und Ergebnis")

    if st.button("▶️ Gravitationskraft berechnen", type="primary"):
        with st.spinner(f"Berechnung läuft..."):
            start_time = time.time()
            if is_prideaux_case:
                force, s_f, s_e, s_v, v, e = calculate_force_prideaux(gauss_n, size1)
                method_used = "Prideaux-Methode"
            else:
                force = calculate_force_direct(gauss_n, size1, size2, gap)
                method_used = "Direkte Integration"
            duration = time.time() - start_time
        st.success(f"Berechnung in {duration:.2f} Sekunden abgeschlossen!")

        # --- Results Display in columns ---
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Simulations-Szene")
            fig_scene = plot_simulation_scene(size1, size2, gap)
            st.pyplot(fig_scene, use_container_width=True)
        with col2:
            st.subheader("🎉 Ergebnis")
            st.metric(label="Berechnete Gravitationskraft F", value=f"{force:.10f}")
            st.subheader("Analyse")
            if is_prideaux_case:
                st.info(f"**Methode:** {method_used}\n\nEs wurde der Spezialfall für identische, berührende Würfel erkannt und die hochpräzise Prideaux-Methode verwendet.")
            else:
                st.info(f"**Methode:** {method_used}\n\nDie Kraft wurde durch direkte Integration berechnet.")
                if gap == 0.0: st.warning("Da sich die Würfel berühren, kann das Ergebnis der direkten Integration ungenau sein.")

        st.markdown("---")

        # --- Detailed Analysis Section (only for the Prideaux case) ---
        if is_prideaux_case:
            with st.expander("Detaillierte Analyse der Prideaux-Methode anzeigen", expanded=True):
                tab_math, tab_flow, tab_decomp = st.tabs(["Mathematische Herleitung", "Berechnungsfluss", "3D-Zerlegung"])
                with tab_math:
                    st.subheader("Schritt-für-Schritt Herleitung")
                    st.markdown("##### 1. S-Werte berechnen"); st.code(f"S_F = {s_f:.8f}\nS_E = {s_e:.8f}\nS_V = {s_v:.8f}", language="text")
                    st.markdown("##### 2. Rekursive Gleichungen anwenden"); st.latex(r"V = \frac{16}{15} S_V"); st.latex(r"E = \frac{V + 8S_E}{7}"); st.latex(r"F = \frac{2E + V + 4S_F}{3}")
                    st.markdown("##### 3. System schrittweise lösen"); st.code(f"V = (16/15)×{s_v:.6f}={v:.8f}\nE=({v:.6f}+8×{s_e:.6f})/7={e:.8f}\nF=(2×{e:.6f}+{v:.6f}+4×{s_f:.6f})/3={force:.8f}", language="text")
                with tab_flow:
                    st.subheader("Visueller Berechnungsfluss"); fig_flow = plot_prideaux_flow(s_f, s_e, s_v, v, e, force); st.pyplot(fig_flow, use_container_width=True)
                with tab_decomp:
                    st.subheader("Visuelle Zerlegung der Würfel"); fig_decomp = plot_prideaux_method_decomposition(size1); st.pyplot(fig_decomp, use_container_width=True)
        else:
            st.info("Für die direkte Integration gibt es keine weitere Zerlegungs-Analyse.")
            
with tab_vergleich:
    st.header("Vergleich der Berechnungsmethoden")
    st.markdown("Die App verwendet je nach Konfiguration zwei fundamental unterschiedliche Methoden, um die Gravitationskraft zu berechnen.")
    
    fig_comp = plot_method_comparison()
    st.pyplot(fig_comp, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Direkte Integration")
        st.info("""
        - **Wann:** Immer wenn die Würfel einen **Abstand** haben oder **unterschiedliche Größen** besitzen.
        - **Wie:** Die Anziehungskraft zwischen Millionen von winzigen Punktpaaren in den beiden Würfeln wird angenähert und aufsummiert.
        - **Vorteil:** Universell einsetzbar für jede Geometrie ohne Berührung.
        - **Nachteil:** Bei Berührung (Abstand = 0) wird diese Methode ungenau, da der Abstand zwischen Punkten null werden kann (Singularität).
        """)
    with col2:
        st.subheader("Prideaux-Methode")
        st.info("""
        - **Wann:** Nur im Spezialfall, wenn zwei **identische Würfel** sich **direkt berühren**.
        - **Wie:** Ein mathematischer Trick. Die Würfel werden in 8 Teilwürfel zerlegt. Anstatt die problematischen, sich berührenden Paare zu berechnen, werden nur die Kräfte der einfach zu berechnenden, getrennten Paare summiert. Über eine Rekursionsformel wird daraus die exakte Gesamtkraft rekonstruiert.
        - **Vorteil:** Umgeht die Singularität elegant und liefert ein extrem präzises Ergebnis.
        - **Nachteil:** Funktioniert nur für diesen einen Spezialfall.
        """)