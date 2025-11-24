# AQI_prediction
# Prédiction de l’Indice de Qualité de l’Air (AQI) à partir des données météorologiques – Évaluation comparative des modèles de machine Learning ( Régression linéaire, Random Forest, XGBoost et SVR)

Ce projet vise à prédire en temps réel l’**Indice de Qualité de l’Air (AQI)** à partir de données atmosphériques issues de capteurs chimiques et de variables météorologiques.  
Il combine une **analyse complète du dataset**, plusieurs **modèles de machine learning**, et une **application Streamlit interactive** permettant de réaliser des prédictions instantanées.
L’étude et l’application reposent notamment sur le **modèle Random Forest**, identifié comme le plus performant avec un **R² ≈ 0.91** et un **MAPE < 10 %**.
## Structure du projet

```
├── data/
│   └── AirQualityUCI.csv
├── notebook/
│   └── Untitled-1.ipynb
├── app/
│   ├── app.py
│   ├── best_model.pkl
│   └── features.pkl
├── README.md
└── requirements.txt
```

---
# Dataset

Le projet utilise le dataset **Air Quality Dataset (UCI Repository)**, contenant des mesures horaires collectées en Italie entre **mars 2004 et avril 2005**.

### Variables principales :
- Polluants : CO, NOx, NO₂, Benzène, NMHC, O₃  
- Capteurs chimiques : PT08.S1, S2, S3, S4, S5  
- Météo : Température (T), Humidité relative (RH)  
- Indice AQI (cible), construit à partir de CO et NO₂

---
# Prétraitement des données

### Nettoyage
- Suppression des colonnes inutiles  
- Remplacement des valeurs aberrantes `-200` par NaN

### Imputation
- **Interpolation temporelle linéaire** adaptée à une série chronologique  
- Propagation avant/arrière si nécessaire

### Feature engineering
- Construction de l’AQI  
- Extraction : heure, jour, mois, saison  
- Normalisation des variables  
- Création de ratios (ex. CO/NO₂)

---

# Modèles de Machine Learning testés

| Modèle | Performance | Commentaire |
|--------|-------------|-------------|
| **Régression linéaire** | R² faible | Sous-estime les valeurs élevées |
| **Random Forest** | **🏆 R² ≈ 0.91, MAE faible** | Le plus robuste, capture les non-linéarités |
| **XGBoost** | Bon, mais < RF | Sensible aux hyperparamètres |
| **SVR (RBF)** | Correct | Bonne modélisation non linéaire |

Le **Random Forest optimisé via GridSearchCV** est retenu comme modèle final.

---

# Résultats principaux

### Performances du modèle Random Forest optimisé 
- **R² : 0.913**  
- **RMSE faible**, **MAPE < 10 %**  
- Importance des variables dominée par : CO, NO₂, PT08.S1, PT08.S2  

### Analyses complémentaires
- Comparaison AQI réel vs prédit  
- Analyse temporelle à 24h (pics matinal et soir)  
- Classification des niveaux AQI (Excellent → Dangereux)  
- Matrice de confusion & F1-score pour validation catégorielle

# Application Streamlit

Une **application interactive Streamlit** utilise le meilleur modèle (Random Forestoptimisé) pour prédire l’AQI en temps réel.

## Fonctionnalités :
### 2 modes :
1. **Mode modèle du notebook** → utilise `best_model.pkl` + `features.pkl`  
2. **Mode de secours CO/NO₂** → interpolation EPA si le modèle n’est pas disponible  

### Interface utilisateur :
- Champs numériques pour saisir CO, NO₂, NMHC, O₃, température, humidité…  
- Affichage :
  - Valeur AQI  
  - Catégorie (Excellent, Bon, Moyen, Médiocre, Dangereux)  
  - Indicateur coloré  
  - Historique des prédictions

---

# Installation & Exécution

### Prérequis
Python 3.10+ (Python 3.13 utilisé)  
pip  
virtualenv (optionnel)

### Installation

```
git clone https://github.com/ton-utilisateur/nom-du-projet.git
cd nom-du-projet
pip install -r requirements.txt
```

### Exécuter le notebook
```
jupyter notebook notebook/Untitled-1.ipynb
```

### Lancer l’application Streamlit
```
cd app
streamlit run app.py
```

---

#  Technologies utilisées

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib / Seaborn  
- Streamlit  
- Jupyter Notebook  

---

# Améliorations possibles

- Connexion à des capteurs IoT (Raspberry Pi, Arduino)  
- Ajout de données temps réel (API AirNow, OpenAQ)  
- Intégration d’un modèle deep learning (LSTM pour séries temporelles)  
- Déploiement cloud (Streamlit Cloud, HuggingFace Spaces)  

---

# Auteur

Projet réalisé par **Marius Bonane**  


