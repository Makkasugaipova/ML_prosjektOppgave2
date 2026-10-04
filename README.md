# Diabetesrisiko-estimator

Prosjektoppgave i **DAT158 Maskinlæring** (ML assignment 2), Høgskulen på Vestlandet, høsten 2026.
Laget av: Makka Sugaipova

En nettside der brukeren legger inn åtte helsemålinger og får et estimat av sannsynligheten for diabetes,
beregnet av en maskinlæringsmodell (logistisk regresjon) trent på *Pima Indians Diabetes Database*.

> ⚠️ Dette er et studentprosjekt og **ikke** et diagnoseverktøy.

## 🔗 Nettsiden
**Live:** [LIM INN LENKE TIL HUGGING FACE SPACE HER]

## 📁 Innhold i repoet

| Mappe / fil | Innhold |
|---|---|
| `data/diabetes.csv` | Datasettet (768 pasienter, 8 features + `Outcome`) |
| `notebooks/01_utforsking_og_baseline.ipynb` | Utforskende dataanalyse (EDA) og baseline |
| `notebooks/02_modellering.ipynb` | Rensing, modellsammenligning, tuning, terskelvalg, evaluering, feature importance og lagring av modell |
| `model/diabetes_model.joblib` | Den ferdigtrente modellen (pipeline + valgt terskel) |
| `app/app.py` | Gradio-nettsiden |
| `report/` | Rapporten |
| `requirements.txt` | Python-pakker som trengs |

## 📊 Resultater (testsett, 154 pasienter)

| Modell | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Baseline: alltid «ikke diabetes» | 0,649 | 0,000 | 0,000 | 0,000 | – |
| Baseline: Glucose ≥ 125 | 0,682 | 0,540 | 0,630 | 0,581 | – |
| **Logistisk regresjon (terskel 0,47)** | **0,714** | **0,571** | **0,741** | **0,645** | **0,810** |

## ▶️ Slik reproduserer du resultatene

Krever Python 3.11 eller nyere.

```bash
# 1. Klon repoet
git clone <lenke-til-dette-repoet>
cd ML_prosjektOppgave2

# 2. Lag og aktiver et virtuelt miljø
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 3. Installer pakkene
pip install -r requirements.txt
```

4. Kjør `notebooks/01_utforsking_og_baseline.ipynb` og deretter `notebooks/02_modellering.ipynb`.
   Notebook 02 lagrer modellen til `model/diabetes_model.joblib`.
5. Start nettsiden lokalt:
   ```bash
   python app/app.py
   ```
   og åpne http://127.0.0.1:7860 i nettleseren.

## 🚀 Deploy på Hugging Face Spaces
1. Lag et nytt Space med SDK **Gradio**.
2. Last opp `app/app.py` (som `app.py` i roten) og mappen `model/`.
3. Legg til en `requirements.txt` med:
   ```
   scikit-learn==1.9.1
   pandas
   joblib
   ```

## 📚 Kilder
- Datasett: UCI Machine Learning (2016). *Pima Indians Diabetes Database*. Kaggle.
  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Smith, J. W. et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.
- Biblioteker: scikit-learn, pandas, matplotlib, seaborn, Gradio.
- KI-verktøyet Claude (Anthropic) er brukt som støtte til oppsett, kode og tekstutkast. Se rapporten for detaljer.
