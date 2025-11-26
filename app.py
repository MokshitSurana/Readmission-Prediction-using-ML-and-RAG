"""
FastAPI Backend for ICU Readmission Risk Prediction (Phase 7)
Integrates cohort-based ML models, FAISS retrieval, and LLM explanations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import numpy as np
import pandas as pd
import joblib
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import re
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# FASTAPI APP INITIALIZATION
# ============================================================
app = FastAPI(title="ICU Readmission Predictor API", version="7.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBAL VARIABLES
# ============================================================
scaler = None
structured_cols = None
trained_specialist_models = {}
faiss_index = None
bert_model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df = None 

# UPDATED WEIGHTS (PHASE 7)
ALPHA_ML = 0.5
BETA_TEXT = 0.5
TARGET_COL = "READMIT_30D"

# ============================================================
# PYDANTIC MODELS
# ============================================================
class PatientData(BaseModel):
    # Demographics
    age: float
    gender: str  # "Male" or "Female"
    ethnicityEncoded: int = 3
    
    # Stay Info
    hospSeq: int = 1
    losHospital: float = 7.0
    losIcu: float = 4.0
    isFirstStay: bool = True
    hospitalExpireFlag: bool = False
    
    # Diagnosis
    diseaseName: str
    dischargeSummary: Optional[str] = None

    # Vitals (First Day Mean)
    heartRate: float
    respiratoryRate: float
    temperature: float
    map: float
    pao2Fio2: float
    urineOutput: float

    # Labs
    wbcCount: float
    plateletCount: float
    creatinine: float
    bilirubin: float
    bun: float
    glucose: float

    # Neuro
    gcsVerbal: float = 5.0
    
    # Severity Scores
    sofa: float
    apsiii: float
    oasis: float
    elixhauserSID30: float
    apsiiiProb: float = 0.1
    elixhauserQuan: float = 0.0

class SimilarCase(BaseModel):
    rank: int
    similarity: float
    readmit: int
    age: float
    sofa: float
    apsiii: float
    elixhauser: float
    text: str
    disease: str

class ClinicalScores(BaseModel):
    sofa: float
    apsiii: float
    oasis: float
    elixhauserSID30: float

class PredictionResponse(BaseModel):
    willBeReadmitted: bool
    probability: float
    mlProbability: float
    textProbability: float
    riskCategory: str
    cohort: str
    reasoning: str
    scores: ClinicalScores
    similarCases: List[SimilarCase]

# ============================================================
# STARTUP: LOAD MODELS
# ============================================================
@app.on_event("startup")
async def load_models():
    global scaler, structured_cols, trained_specialist_models
    global faiss_index, bert_model, tokenizer, train_df
    
    logger.info("🚀 Loading Phase 7 models and artifacts...")
    
    try:
        # 1. Load Scaler & Cols
        scaler = joblib.load("structured_scaler.pkl")
        structured_cols = joblib.load("structured_cols.pkl")
        
        # 2. Load Cohort Models (Updated Filenames)
        model_files = {
            "SEPTICEMIA": "septicemia_cohort_catboost.pkl",
            "CORONARY_ATHEROSCLEROSIS": "coronary_atherosclerosis_cohort_randomforest.pkl",
            "RESPIRATORY_FAILURE": "respiratory_failure_cohort_randomforest.pkl",
            "SUBENDOCARDIAL_INFARCTION": "subendocardial_infarction_cohort_randomforest.pkl",
            "AORTIC_VALVE_DISORDER": "aortic_valve_disorder_cohort_catboost.pkl",
            "GENERAL_MEDICINE": "general_model_xgboost.pkl",
        }
        
        for cohort, filename in model_files.items():
            try:
                trained_specialist_models[cohort] = joblib.load(filename)
                logger.info(f"✅ Loaded {cohort}")
            except FileNotFoundError:
                logger.warning(f"⚠️ Missing: {filename}")
        
        # 3. Load FAISS & Data
        faiss_index = faiss.read_index("faiss_bert_train.index")
        train_df = pd.read_csv("train_df_with_text.csv")
        
        # 4. Load BERT
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name).to(device)
        bert_model.eval()
        
        logger.info("🎉 System ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup Failed: {str(e)}")
        raise

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_patient_cohort(disease_text: str) -> str:
    t = str(disease_text).upper()
    if "SEPTIC" in t: return "SEPTICEMIA"
    if "CORONARY" in t or "ATHEROSCL" in t: return "CORONARY_ATHEROSCLEROSIS"
    if "RESPIRATORY" in t or "FAILURE" in t: return "RESPIRATORY_FAILURE"
    if "SUBENDOCARDIAL" in t: return "SUBENDOCARDIAL_INFARCTION"
    if "AORTIC" in t or "VALVE" in t: return "AORTIC_VALVE_DISORDER"
    return "GENERAL_MEDICINE"

def get_risk_label(prob: float) -> str:
    if prob < 0.10: return "No Risk"
    elif prob < 0.30: return "Low Risk"
    elif prob < 0.60: return "Moderate Risk"
    else: return "High Risk"

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = bert_model(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy().astype("float32")

def retrieve_similar_cases(text: str, k: int = 3) -> List[Dict]:
    vec = get_bert_embedding(text).reshape(1, -1)
    dist, idxs = faiss_index.search(vec, k)
    
    results = []
    for rank, (d, idx) in enumerate(zip(dist[0], idxs[0]), start=1):
        row = train_df.iloc[idx]
        results.append({
            "rank": rank,
            "similarity": float(d),
            "readmit": int(row.get(TARGET_COL, 0)),
            "age": float(row.get("admission_age", 0)),
            "sofa": float(row.get("SOFA", 0)),
            "apsiii": float(row.get("apsiii", 0)),
            "elixhauser": float(row.get("elixhauser_SID30", 0)),
            "text": str(row.get("TEXT", ""))[:300],
            "disease": str(row.get("patient_disease", ""))
        })
    return results

def query_llm(prompt: str) -> str:
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': 'mistral', 'prompt': prompt, 'stream': False},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()['response']
        return "LLM unavailable."
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return "LLM unavailable."

def parse_llm_risk(output: str, default: float = 0.5) -> float:
    try:
        match = re.search(r"RISK\s*=\s*([0-1]\.\d+)", output)
        if match:
            return float(np.clip(float(match.group(1)), 0.0, 1.0))
    except:
        pass
    return default

# ============================================================
# PROMPT BUILDERS
# ============================================================
def build_scoring_prompt(patient_dict, cases, cohort) -> str:
    sex = patient_dict.get('gender', 'Unknown')
    p_block = f"""
New patient (cohort: {cohort}):
- Age: {patient_dict.get('age')} | Sex: {sex}
- Diagnosis: {patient_dict.get('diseaseName')}
- SOFA: {patient_dict.get('sofa')} | APS III: {patient_dict.get('apsiii')} | Elixhauser: {patient_dict.get('elixhauserSID30')}
- Note: {patient_dict.get('dischargeSummary', '')[:500]}
"""
    c_blocks = []
    for c in cases:
        c_blocks.append(f"""
Case {c['rank']}: Readmit={c['readmit']} | Age={c['age']:.0f} | SOFA={c['sofa']:.1f} | APSIII={c['apsiii']:.1f}
Note: {c['text']}...
""")
    
    return f"""
You are an ICU readmission expert.
Estimate 30-day readmission risk (0 to 1) for the new patient based on similar cases.

{p_block}
Similar Cases:
{''.join(c_blocks)}

Output ONLY:
RISK=0.xx|NOTE=reason
"""

def build_explanation_prompt(patient_dict, cases, fusion_prob, cohort) -> str:
    sex = patient_dict.get('gender', 'Unknown')
    label = get_risk_label(fusion_prob)
    
    p_block = f"""
**Patient Info**
- Age: {patient_dict.get('age')} | Sex: {sex}
- Diagnosis: {patient_dict.get('diseaseName')}
- SOFA: {patient_dict.get('sofa')} | APS III: {patient_dict.get('apsiii')} | Elixhauser: {patient_dict.get('elixhauserSID30')}
- Summary: {patient_dict.get('dischargeSummary', '')}
"""
    c_blocks = []
    for c in cases:
        c_blocks.append(f"""
**Case {c['rank']}** (Outcome: {'Readmitted' if c['readmit'] else 'No Readmission'})
- Age: {c['age']:.0f}, SOFA: {c['sofa']:.1f}, APSIII: {c['apsiii']:.1f}
- Summary: {c['text']}...
""")

    # UPDATED PROMPT: Explicitly instructs LLM to avoid specific case references
    return f"""
You are a clinical assistant.
The model predicted: **{fusion_prob*100:.1f}% ({label})**.

Explain this risk using the data below.

{p_block}
**Similar Historical Cases (For Context Only - DO NOT CITE SPECIFIC CASE NUMBERS):**
{''.join(c_blocks)}

REQUIRED FORMAT:

**Patient Summary:**
<1-2 sentences>

**Prediction:**
Estimated readmission probability: **{fusion_prob*100:.1f}% ({label})**

**Justification:**
<1-3 sentences explaining risk based on severity scores and clinical factors.>
<CRITICAL: The user cannot see the similar cases. Do NOT refer to "Case 1" or "Case 2" by name. Instead, make general comparisons like "compared to similar historical patients...">
"""

# ============================================================
# PREDICTION ENDPOINT
# ============================================================
@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    try:
        logger.info(f"🔮 Predicting for: {patient.diseaseName}")
        
        # 1. BASE PATIENT SAMPLING (FIX FOR 100% RISK BUG)
        # We sample a real row to fill in missing background vitals (HR, Temp, etc)
        base_row = train_df.sample(1, random_state=123).iloc[0].copy()
        
        # 2. MAP FRONTEND INPUTS TO BACKEND COLUMNS
        # Define user inputs with keys matching Phase 7 requirements
        user_inputs = {
            "admission_age": patient.age,
            "gender_encoded": 1 if patient.gender.lower() == "male" else 0,
            "ethnicity_grouped_encoded": patient.ethnicityEncoded,
            "hospstay_seq": patient.hospSeq,
            "los_hospital": patient.losHospital,
            "los_icu": patient.losIcu,
            "first_hosp_stay_encoded": 1 if patient.isFirstStay else 0,
            
            # Vitals
            "vitals_first_day__heartrate__mean": patient.heartRate,
            "tempc": patient.temperature,
            "meanbp": patient.map,
            "resprate": patient.respiratoryRate,
            "pao2fio2ratio_vent": patient.pao2Fio2,
            "urineoutput": patient.urineOutput,
            
            # Labs
            "wbc": patient.wbcCount,
            "creatinine": patient.creatinine,
            "platelet": patient.plateletCount,
            "bilirubin": patient.bilirubin,
            "bun": patient.bun,
            "glucose": patient.glucose,
            
            # Neuro
            "gcsverbal": patient.gcsVerbal,
            
            # Scores
            "sofa": patient.sofa,
            "apsiii": patient.apsiii,
            "oasis__oasis__mean": patient.oasis,
            "elixhauser_sid30": patient.elixhauserSID30,
            "apsiii_prob": patient.apsiiiProb,
            "elixhauser_quan_score__elixhauser_vanwalraven__mean": patient.elixhauserQuan,
            
            "hospital_expire_flag": 1 if patient.hospitalExpireFlag else 0
        }

        # 3. MERGE INPUTS INTO DATAFRAME
        df_input = pd.DataFrame([base_row[structured_cols].values], columns=structured_cols)
        
        # Robust Case-Insensitive Mapping
        for model_col in structured_cols:
            for u_key, u_val in user_inputs.items():
                if model_col.lower() == u_key.lower():
                    df_input[model_col] = u_val
                    break

        # 4. STRUCTURED ML PREDICTION
        cohort = get_patient_cohort(patient.diseaseName)
        model = trained_specialist_models.get(cohort, trained_specialist_models["GENERAL_MEDICINE"])
        
        x_scaled = scaler.transform(df_input)
        ml_prob = float(model.predict_proba(x_scaled)[0][1])
        logger.info(f"🤖 ML Prob: {ml_prob:.4f}")

        # 5. RETRIEVAL & LLM
        clean_text = (patient.dischargeSummary or patient.diseaseName).strip()
        similar_cases = retrieve_similar_cases(clean_text, k=3)
        
        # Build prompt dict
        patient_dict = patient.dict()
        patient_dict['gender'] = "Male" if user_inputs['gender_encoded'] == 1 else "Female"
        
        # LLM Scoring
        score_prompt = build_scoring_prompt(patient_dict, similar_cases, cohort)
        llm_raw = query_llm(score_prompt)
        text_prob = parse_llm_risk(llm_raw, default=0.5)
        logger.info(f"📝 Text Prob: {text_prob:.4f}")

        # 6. FUSION
        final_prob = ALPHA_ML * ml_prob + BETA_TEXT * text_prob
        risk_cat = get_risk_label(final_prob)
        
        # 7. EXPLANATION
        explain_prompt = build_explanation_prompt(patient_dict, similar_cases, final_prob, cohort)
        reasoning_output = query_llm(explain_prompt)
        
        # Cleanup reasoning text
        try:
            if "Justification:" in reasoning_output:
                reasoning = reasoning_output.split("Justification:")[1].strip()
            else:
                reasoning = reasoning_output
        except:
            reasoning = reasoning_output

        # 8. RESPONSE
        return PredictionResponse(
            willBeReadmitted=(final_prob >= 0.30),
            probability=round(final_prob, 4),
            mlProbability=round(ml_prob, 4),
            textProbability=round(text_prob, 4),
            riskCategory=risk_cat,
            cohort=cohort,
            reasoning=reasoning,
            scores=ClinicalScores(
                sofa=patient.sofa,
                apsiii=patient.apsiii,
                oasis=patient.oasis,
                elixhauserSID30=patient.elixhauserSID30
            ),
            similarCases=[SimilarCase(**c) for c in similar_cases]
        )

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)