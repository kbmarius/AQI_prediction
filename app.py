# -*- coding: utf-8 -*-
"""
App Streamlit : Prédiction d'AQI en temps réel
------------------------------------------------
- Charge un modèle entraîné (best_model.pkl) + liste de features (features.pkl) si présents
- Sinon, bascule en mode "secours" (règles EPA à partir de CO & NO2)
- Affiche l'AQI prédit, la classe correspondante et un indicateur visuel

Pour lancer :
    pip install -r requirements.txt
    streamlit run app.py
"""
import os
import time
import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except Exception:  # joblib peut être absent ; l'app fonctionne quand même en mode secours
    joblib = None

# -----------------------------
# 1) Seuils & utilitaires AQI
# -----------------------------

def classify_aqi(aqi_value: float) -> str:
    """Seuils identiques au notebook (Excellent/Bon/Moyen/Médiocre/Dangereux)."""
    if aqi_value <= 25:
        return "Excellent"
    elif aqi_value <= 50:
        return "Bon"
    elif aqi_value <= 75:
        return "Moyen"
    elif aqi_value <= 100:
        return "Médiocre"
    else:
        return "Dangereux"

CLASS_COLORS = {
    "Excellent": "#2ECC71",  # vert
    "Bon":       "#A3E4D7",  # vert clair
    "Moyen":     "#F1C40F",  # jaune
    "Médiocre":  "#E67E22",  # orange
    "Dangereux": "#E74C3C",  # rouge
}

# Conversions (≈ 25°C, 1 atm)
def co_mg_m3_to_ppm(co_mg_m3):
    return np.nan if pd.isna(co_mg_m3) else co_mg_m3 * 0.873

def no2_ug_m3_to_ppb(no2_ug_m3):
    return np.nan if pd.isna(no2_ug_m3) else no2_ug_m3 * 0.5229

# Règles EPA (interpolation linéaire)
CO_BREAKS = [
    (0.0, 4.4,   0,  50),
    (4.5, 9.4,  51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4,151, 200),
    (15.5, 30.4,201, 300),
    (30.5, 40.4,301, 400),
    (40.5, 50.4,401, 500),
]

NO2_BREAKS = [
    (0,   53,    0,  50),
    (54,  100,  51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249,201, 300),
    (1250,1649,301, 400),
    (1650,2049,401, 500),
]

def linear_aqi(Cp, Clow, Chigh, Ilow, Ihigh):
    return (Ihigh - Ilow) / (Chigh - Clow) * (Cp - Clow) + Ilow

def aqi_from_breaks(Cp, breaks):
    if pd.isna(Cp):
        return np.nan
    for Clow, Chigh, Ilow, Ihigh in breaks:
        if Clow <= Cp <= Chigh:
            return linear_aqi(Cp, Clow, Chigh, Ilow, Ihigh)
    max_C = max(b[1] for b in breaks)
    if Cp > max_C:
        return 500.0
    return np.nan

def aqi_epa_from_co_no2(co_mg_m3, no2_ug_m3):
    co_ppm_approx_8h = co_mg_m3_to_ppm(co_mg_m3)
    no2_ppb_1h = no2_ug_m3_to_ppb(no2_ug_m3)
    sub_co = aqi_from_breaks(co_ppm_approx_8h, CO_BREAKS)
    sub_no2 = aqi_from_breaks(no2_ppb_1h, NO2_BREAKS)
    return float(np.nanmax([sub_co, sub_no2]))

# -----------------------------
# 2) Chargement du modèle
# -----------------------------
MODEL_PATH = "best_model.pkl"
FEATS_PATH = "features.pkl"

model = None
features = None
model_status = "Aucun modèle trouvé — mode secours disponible"

if os.path.exists(MODEL_PATH) and os.path.exists(FEATS_PATH) and joblib is not None:
    try:
        model = joblib.load(MODEL_PATH)
        feats = joblib.load(FEATS_PATH)
        if isinstance(feats, (list, tuple, np.ndarray)):
            features = list(feats)
        elif isinstance(feats, dict) and "features" in feats:
            features = list(feats["features"])
        else:
            features = list(feats)
        model_status = "Modèle chargé ✓"
    except Exception as e:
        model = None
        features = None
        model_status = f"Échec chargement modèle : {e}"

# -----------------------------
# 3) UI Streamlit
# -----------------------------
st.set_page_config(page_title="AQI Temps Réel", page_icon="🌫️", layout="centered")
st.title("🌫️ Prédiction d’AQI en temps réel")
st.caption("S'appuie sur ton modèle du notebook s'il est présent ; sinon, calcul de secours via CO & NO₂.")

st.sidebar.header("Configuration")
st.sidebar.info(model_status)
mode = st.sidebar.radio(
    "Mode", ["Modèle du notebook", "Calcul de secours (CO & NO₂)"], index=0 if model is not None else 1
)

def colored_badge(label: str):
    color = CLASS_COLORS.get(label, "#BDC3C7")
    st.markdown(
        f"""
        <div style="display:inline-block;padding:.35rem .6rem;border-radius:999px;background:{color};color:#1B2631;font-weight:600;">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

def aqi_progress(aqi: float, max_ref: float = 150.0):
    pct = int(max(0.0, min(100.0, (aqi / max_ref) * 100.0)))
    try:
        # Streamlit >= 1.27 supporte text=
        st.progress(pct, text=f"AQI ~ {aqi:.1f} (≈{pct}%)")
    except TypeError:
        st.progress(pct)
        st.caption(f"AQI ~ {aqi:.1f} (≈{pct}%)")

st.markdown("---")

# -----------------------------
# 4) Prédiction en direct
# -----------------------------
if mode == "Modèle du notebook" and model is not None and features is not None:
    st.subheader("🧠 Entrées du modèle")
    st.caption("Les champs ci-dessous reprennent les features utilisées à l'entraînement.")

    def default_for(feat: str) -> float:
        f = feat.lower()
        if any(k in f for k in ["temp"]):
            return 20.0
        if any(k in f for k in ["rh", "humid"]):
            return 50.0
        if any(k in f for k in ["co", "no2", "c6h6", "nmhc", "o3", "so2", "pm"]):
            return 1.0
        return 0.0

    input_vals = {}
    cols = st.columns(2)
    for i, feat in enumerate(features):
        with cols[i % 2]:
            input_vals[feat] = st.number_input(feat, value=float(default_for(feat)), step=0.1, format="%.3f")

    X = pd.DataFrame([input_vals]).reindex(columns=features, fill_value=0.0)

    try:
        y_pred = float(model.predict(X)[0])
        aqi_class = classify_aqi(y_pred)

        st.markdown("### 🎯 Résultat (modèle)")
        st.metric("AQI", f"{y_pred:.1f}")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Classe :**")
        with col2:
            colored_badge(aqi_class)
        aqi_progress(y_pred)

        with st.expander("Détails des entrées"):
            st.dataframe(X.T.rename(columns={0: "valeur"}))

        hist = st.session_state.get("history", [])
        hist.append({"timestamp": time.time(), "AQI": y_pred, "classe": aqi_class})
        st.session_state["history"] = hist[-200:]

        if st.checkbox("Afficher l'historique de cette session"):
            hdf = pd.DataFrame(st.session_state["history"])
            if not hdf.empty:
                hdf["timestamp"] = pd.to_datetime(hdf["timestamp"], unit="s")
                st.line_chart(hdf.set_index("timestamp")["AQI"])
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")
        st.info("Vérifie que best_model.pkl encapsule tout le prétraitement (Pipeline) et que features.pkl correspond.")

else:
    st.subheader("🧮 Calcul de secours (EPA-like)")
    st.caption("Utilise CO(GT) [mg/m³] et NO₂(GT) [µg/m³]; AQI = max(sous-indices CO(8h), NO₂(1h).)")

    c1, c2 = st.columns(2)
    with c1:
        co = st.number_input("CO(GT) [mg/m³]", value=1.0, min_value=0.0, step=0.1, format="%.2f")
    with c2:
        no2 = st.number_input("NO₂(GT) [µg/m³]", value=40.0, min_value=0.0, step=1.0, format="%.1f")

    aqi = aqi_epa_from_co_no2(co, no2)
    aqi_class = classify_aqi(aqi)

    st.markdown("### 🎯 Résultat (secours)")
    st.metric("AQI", f"{aqi:.1f}")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Classe :**")
    with col2:
        colored_badge(aqi_class)
    aqi_progress(aqi)

    with st.expander("Détails des conversions"):
        st.write(f"CO ≈ {co_mg_m3_to_ppm(co):.2f} ppm (8h, approx)")
        st.write(f"NO₂ ≈ {no2_ug_m3_to_ppb(no2):.1f} ppb (1h)")

st.markdown("---")
st.caption("💡 Astuce : depuis le notebook, sauvegarde ton modèle : joblib.dump(best_model, 'best_model.pkl') et joblib.dump(features, 'features.pkl').")
