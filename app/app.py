"""
Diabetesrisiko-estimator – Gradio-nettside (DAT158, ML assignment 2)

Laster den trente modellen fra notebook 02 (model/diabetes_model.joblib)
og gir et estimat av sannsynligheten for diabetes basert på 8 helsemålinger.

Kjør lokalt fra prosjektmappen:
    python app/app.py
Åpne deretter http://127.0.0.1:7860 i nettleseren.
"""
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

# Finn modellfilen både lokalt (../model) og på Hugging Face (model/ ved siden av app.py)
HERE = Path(__file__).resolve().parent
CANDIDATES = [
    HERE / "model" / "diabetes_model.joblib",
    HERE.parent / "model" / "diabetes_model.joblib",
]
MODEL_PATH = next((p for p in CANDIDATES if p.exists()), None)
if MODEL_PATH is None:
    raise FileNotFoundError(
        "Fant ikke diabetes_model.joblib. Kjør notebook 02 først, "
        "eller legg modellen i en mappe som heter 'model'."
    )

bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
THRESHOLD = bundle["threshold"]
FEATURES = bundle["features"]


def predict(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    # Tomme felt tolkes som "ukjent" (0), slik som i treningsdataene
    values = {k: (0 if v is None else float(v)) for k, v in values.items()}
    X = pd.DataFrame([values])[FEATURES]

    p = float(MODEL.predict_proba(X)[0, 1])
    high = p >= THRESHOLD

    if high:
        verdict = "## 🔴 Forhøyet risiko"
        advice = (
            "Modellen vurderer risikoen som **forhøyet**. "
            "Det anbefales å kontakte lege for en blodprøve (f.eks. HbA1c)."
        )
    else:
        verdict = "## 🟢 Lav risiko"
        advice = (
            "Modellen vurderer risikoen som **lav**. "
            "Har du symptomer eller er bekymret, bør du likevel snakke med lege."
        )

    details = (
        f"**Estimert sannsynlighet for diabetes:** {p:.0%}  \n"
        f"Grensen for «forhøyet risiko» er satt til {THRESHOLD:.0%}. "
        "Den er bevisst satt lavt for å fange opp flest mulig som bør testes."
    )
    missing = [k for k in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"] if values[k] == 0]
    if missing:
        details += (
            "  \n_Følgende verdier ble tolket som ukjente og erstattet med typiske verdier: "
            + ", ".join(missing) + "._"
        )

    return f"{verdict}\n\n{advice}\n\n{details}"


DISCLAIMER = """
> ⚠️ **Dette er ikke en diagnose.** Verktøyet er laget som et studentprosjekt (DAT158, HVL) og gir kun et
> statistisk estimat. Modellen er trent på data fra kvinner over 21 år med Pima-indiansk bakgrunn
> (Pima Indians Diabetes Database), og resultatene er ikke nødvendigvis gyldige for andre grupper.
> Ingen data du legger inn blir lagret.
"""

with gr.Blocks(title="Diabetesrisiko-estimator") as demo:
    gr.Markdown(
        "# 🩺 Diabetesrisiko-estimator\n"
        "Legg inn helsemålingene nedenfor og trykk **Beregn risiko**. "
        "En maskinlæringsmodell (logistisk regresjon) estimerer sannsynligheten for diabetes."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            glucose = gr.Number(label="Blodsukker (mg/dL)",
                                info="Plasmaglukose 2 timer etter glukosebelastning. 0 = ukjent", value=120)
            bmi = gr.Number(label="BMI (kg/m²)", info="Vekt (kg) / høyde² (m). 0 = ukjent", value=30.0)
            age = gr.Number(label="Alder (år)", value=35, precision=0)
            pregnancies = gr.Number(label="Antall graviditeter", value=1, precision=0)
        with gr.Column():
            blood_pressure = gr.Number(label="Diastolisk blodtrykk (mm Hg)", info="0 = ukjent", value=70)
            dpf = gr.Number(label="Arvelig disposisjon (Diabetes Pedigree Function)",
                            info="Typisk 0,1–2,5. Høyere = mer diabetes i familien", value=0.4)
            insulin = gr.Number(label="Insulin (µU/ml)", info="Etter 2 timer. 0 = ukjent", value=0)
            skin_thickness = gr.Number(label="Hudfoldtykkelse triceps (mm)", info="0 = ukjent", value=0)

    button = gr.Button("Beregn risiko", variant="primary")
    output = gr.Markdown()

    inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    button.click(predict, inputs=inputs, outputs=output)

    gr.Examples(
        label="Eksempler (klikk for å fylle inn)",
        examples=[
            [6, 148, 72, 35, 0, 33.6, 0.627, 50],
            [1, 85, 66, 29, 0, 26.6, 0.351, 31],
            [0, 137, 40, 35, 168, 43.1, 2.288, 33],
        ],
        inputs=inputs,
    )

if __name__ == "__main__":
    demo.launch()
