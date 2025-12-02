from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import google.generativeai as genai
import os
import re
from typing import Dict, Any, Union, List

from fastapi.middleware.cors import CORSMiddleware

# ---------------- Initialize ----------------
app = FastAPI(title="LLM Model API", version="3.4")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ Load environment variables
load_dotenv()

# ✅ Fetch Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Please set it in your .env or environment variables.")

# ✅ Configure Gemini Client
genai.configure(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"


# ---------------- Schema ----------------
class BiomarkerRequest(BaseModel):
    # ---------------- Patient Info ----------------
    id: str = Field(default="PT01", description="ID For Patient")
    age: int = Field(default=52, description="Patient age in years")
    gender: str = Field(default="female", description="Gender of the patient")
    height: float = Field(default=165, description="Height in cm")
    weight: float = Field(default=70, description="Weight in kg")

    # ---------------- Kidney Function ----------------
    urea: float = Field(default=30.0, description="Urea (S) in mg/dL")
    creatinine: float = Field(default=1.0, description="Creatinine (S) in mg/dL")
    uric_acid: float = Field(default=5.0, description="Uric Acid (S) in mg/dL")
    calcium: float = Field(default=9.5, description="Calcium (S) in mg/dL")
    phosphorus: float = Field(default=3.5, description="Phosphorus (S) in mg/dL")
    sodium: float = Field(default=140.0, description="Sodium (S) in mEq/L")
    potassium: float = Field(default=4.2, description="Potassium (S) in mEq/L")
    chloride: float = Field(default=102.0, description="Chloride (S) in mEq/L")
    amylase: float = Field(default=70.0, description="Amylase (S) in U/L")
    lipase: float = Field(default=35.0, description="Lipase (S) in U/L")
    bicarbonate: float = Field(default=24.0, description="Bicarbonate (S) in mEq/L")
    egfr: float = Field(default=100.0, description="Estimated GFR (S) in mL/min/1.73m²")
    serum_osmolality: float = Field(default=290.0, description="Serum Osmolality (S) in mOsm/kg")
    ionized_calcium: float = Field(default=1.25, description="Ionized Calcium (S) in mmol/L")
    
    # ---------------- Basic Check-up ----------------
    wbc: float = Field(default=6.0, description="White Blood Cell count (×10^3/μL)")
    hemoglobin: float = Field(default=14.0, description="Hemoglobin (g/dL)")
    mcv: float = Field(default=90.0, description="Mean Corpuscular Volume (fL)")
    rdw: float = Field(default=13.5, description="Red Cell Distribution Width (%)")
    lymphocytes: float = Field(default=30.0, description="Lymphocyte percentage (%)")
    
    # ---------------- Diabetic Profile ----------------
    fasting_blood_sugar: float = Field(default=85.0, description="Fasting Blood Sugar (mg/dL)")
    hb1ac: float = Field(default=5.4, description="HbA1c (%)")
    insulin: float = Field(default=10.0, description="Insulin (µIU/mL)")
    c_peptide: float = Field(default=1.2, description="C-Peptide (ng/mL)")
    homa_ir: float = Field(default=1.2, description="HOMA-IR")
    
    # ---------------- Lipid Profile ----------------
    total_cholesterol: float = Field(default=180.0, description="Total Cholesterol (mg/dL)")
    ldl: float = Field(default=90.0, description="LDL Cholesterol (mg/dL)")
    hdl: float = Field(default=50.0, description="HDL Direct (mg/dL)")
    cholesterol_hdl_ratio: float = Field(default=3.0, description="Cholesterol/HDL Ratio")
    triglycerides: float = Field(default=120.0, description="Triglycerides (mg/dL)")
    apo_a1: float = Field(default=140.0, description="Apo A-1 (mg/dL)")
    apo_b: float = Field(default=70.0, description="Apo B (mg/dL)")
    apo_ratio: float = Field(default=0.5, description="Apo B : Apo A-1 ratio")
    
    # ---------------- Liver Function ----------------
    albumin: float = Field(default=4.2, description="Albumin (g/dL)")
    total_protein: float = Field(default=7.0, description="Total Protein (g/dL)")
    alt: float = Field(default=25.0, description="ALT (U/L)")
    ast: float = Field(default=24.0, description="AST (U/L)")
    alp: float = Field(default=120.0, description="ALP (U/L)")
    ggt: float = Field(default=20.0, description="GGT (U/L)")
    ld: float = Field(default=180.0, description="LDH (U/L)")
    globulin: float = Field(default=3.0, description="Globulin (g/dL)")
    albumin_globulin_ratio: float = Field(default=1.4, description="Albumin/Globulin Ratio")
    magnesium: float = Field(default=2.0, description="Magnesium (mg/dL)")
    total_bilirubin: float = Field(default=0.7, description="Total Bilirubin (mg/dL)")
    direct_bilirubin: float = Field(default=0.3, description="Direct Bilirubin (mg/dL)")
    indirect_bilirubin: float = Field(default=0.4, description="Indirect Bilirubin (mg/dL)")
    ammonia: float = Field(default=35.0, description="Ammonia (NH3) (µmol/L)")
    
    # ---------------- Cardiac Profile ----------------
    hs_crp: float = Field(default=1.0, description="High-Sensitivity CRP (mg/L)")
    ck: float = Field(default=150.0, description="Creatine Kinase (U/L)")
    ck_mb: float = Field(default=20.0, description="CK-MB (U/L)")
    homocysteine: float = Field(default=10.0, description="Homocysteine (µmol/L)")
    
    # ---------------- Mineral & Heavy Metal ----------------
    zinc: float = Field(default=90.0, description="Zinc (µg/dL)")
    copper: float = Field(default=100.0, description="Copper (µg/dL)")
    selenium: float = Field(default=120.0, description="Selenium (µg/L)")
    
    # ---------------- Iron Profile ----------------
    iron: float = Field(default=100.0, description="Iron (µg/dL)")
    tibc: float = Field(default=300.0, description="TIBC (µg/dL)")
    transferrin: float = Field(default=250.0, description="Transferrin (mg/dL)")
    
    # ---------------- Vitamins ----------------
    vitamin_d: float = Field(default=35.0, description="Vitamin D (ng/mL)")
    vitamin_b12: float = Field(default=500.0, description="Vitamin B12 (pg/mL)")
    
    # ---------------- Hormone Profile ----------------
    total_testosterone: float = Field(default=450.0, description="Total Testosterone (ng/dL)")
    free_testosterone: float = Field(default=15.0, description="Free Testosterone (pg/mL)")
    estrogen: float = Field(default=60.0, description="Estrogen / Estradiol (pg/mL)")
    progesterone: float = Field(default=1.0, description="Progesterone (ng/mL)")
    dhea_s: float = Field(default=250.0, description="DHEA-S (µg/dL)")
    shbg: float = Field(default=40.0, description="SHBG (nmol/L)")
    lh: float = Field(default=5.0, description="LH (IU/L)")
    fsh: float = Field(default=6.0, description="FSH (IU/L)")
    
    # ---------------- Thyroid Profile ----------------
    tsh: float = Field(default=2.0, description="TSH (µIU/mL)")
    free_t3: float = Field(default=3.2, description="Free T3 (pg/mL)")
    free_t4: float = Field(default=1.2, description="Free T4 (ng/dL)")
    total_t3: float = Field(default=120.0, description="Total T3 (ng/dL)")
    total_t4: float = Field(default=8.0, description="Total T4 (µg/dL)")
    reverse_t3: float = Field(default=15.0, description="Reverse T3 (ng/dL)")
    tpo_ab: float = Field(default=5.0, description="Thyroid Antibodies – TPO Ab (IU/mL)")
    tg_ab: float = Field(default=3.0, description="Thyroid Antibodies – TG Ab (IU/mL)")
    
    # ---------------- Adrenal / Stress / Other Hormones ----------------
    cortisol: float = Field(default=12.0, description="Cortisol (µg/dL)")
    acth: float = Field(default=25.0, description="ACTH (pg/mL)")
    igf1: float = Field(default=200.0, description="IGF-1 (ng/mL)")
    leptin: float = Field(default=10.0, description="Leptin (ng/mL)")
    adiponectin: float = Field(default=10.0, description="Adiponectin (µg/mL)")
    
    # ---------------- Blood Marker Cancer Profile ----------------
    ca125: float = Field(default=20.0, description="CA125 (U/mL)")
    ca15_3: float = Field(default=25.0, description="CA15-3 (U/mL)")
    ca19_9: float = Field(default=30.0, description="CA19-9 (U/mL)")
    psa: float = Field(default=1.0, description="PSA (ng/mL)")
    cea: float = Field(default=2.0, description="CEA (ng/mL)")
    calcitonin: float = Field(default=5.0, description="Calcitonin (pg/mL)")
    afp: float = Field(default=5.0, description="AFP (ng/mL)")
    tnf: float = Field(default=2.0, description="Tumor Necrosis Factor (pg/mL)")
    
    # ---------------- Immune Profile ----------------
    ana: float = Field(default=0.5, description="ANA (IU/mL)")
    ige: float = Field(default=100.0, description="IgE (IU/mL)")
    igg: float = Field(default=1200.0, description="IgG (mg/dL)")
    anti_ccp: float = Field(default=10.0, description="Anti-CCP (U/mL)")
    dsdna: float = Field(default=0.5, description="dsDNA (IU/mL)")
    ssa_ssb: float = Field(default=5.0, description="SSA/SSB (IU/mL)")
    rnp: float = Field(default=1.0, description="RNP (IU/mL)")
    sm_antibodies: float = Field(default=0.5, description="Sm Antibodies (IU/mL)")
    anca: float = Field(default=0.5, description="ANCA (IU/mL)")
    anti_ena: float = Field(default=0.5, description="Anti-ENA (IU/mL)")
    il6: float = Field(default=3.0, description="IL-6 (pg/mL)")
    allergy_panel: float = Field(default=10.0, description="Comprehensive Allergy Profile (IgE & Food Sensitivity IgG)")


ref=dict({
  "urea": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "10-50", "unit": "mg/dL" }
  ],

  "creatinine": [
    { "gender": "male", "age_min": 18, "age_max": 60, "range": "0.7-1.2", "unit": "mg/dL" },
    { "gender": "female", "age_min": 18, "age_max": 60, "range": "0.5-0.9", "unit": "mg/dL" },
    { "gender": "male", "age_min": 60, "age_max": 90, "range": "0.9-1.3", "unit": "mg/dL" },
    { "gender": "female", "age_min": 60, "age_max": 90, "range": "0.6-1.2", "unit": "mg/dL" },
    { "gender": "children", "age_min": 0, "age_max": 17, "range": "0.3-0.7", "unit": "mg/dL" }
  ],

  "uric_acid": [
    { "gender": "male", "age_min": 18, "age_max": 120, "range": "3.5-7.2", "unit": "mg/dL" },
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "2.5-6.2", "unit": "mg/dL" },
    { "gender": "children", "age_min": 0, "age_max": 17, "range": "2-5", "unit": "mg/dL" }
  ],

  "calcium": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "8.6-10.3", "unit": "mg/dL" }
  ],

  "phosphorus": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "2.5-4.5", "unit": "mg/dL" },
    { "gender": "children", "age_min": 0, "age_max": 17, "range": "4-7", "unit": "mg/dL" }
  ],

  "sodium": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "135-145", "unit": "mmol/L" }
  ],

  "potassium": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "3.5-5.3", "unit": "mmol/L" }
  ],

  "chloride": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "90-110", "unit": "mmol/L" }
  ],

  "amylase": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<90", "unit": "U/L" }
  ],

  "lipase": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "10-60", "unit": "U/L" }
  ],

  "bicarbonate": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "22-29", "unit": "mmol/L" }
  ],

  "egfr": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": ">60", "unit": "mL/min/1.73m2" }
  ],

  "serum_osmolality": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "240-321", "unit": "mmol/kg" }
  ],

  "ionized_calcium": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "1.09-1.33", "unit": "mmol/L" }
  ],

  "wbc": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "4.0-11.0", "unit": "10^3/uL" }
  ],

  "hemoglobin": [
    { "gender": "male", "age_min": 18, "age_max": 120, "range": "14-18", "unit": "g/dL" },
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "12-15", "unit": "g/dL" }
  ],

  "mcv": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "80-100", "unit": "fL" }
  ],

  "rdw": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "11.5-14.5", "unit": "%" }
  ],

  "lymphocytes": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "20-40", "unit": "%" }
  ],

  "fasting_blood_sugar": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "<92", "unit": "mg/dL" }
  ],

  "hb1ac": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "4.0-5.6", "unit": "%" }
  ],

  "insulin": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "<17", "unit": "uIU/mL" },
    { "gender": "children", "age_min": 0, "age_max": 17, "range": "1-17.5", "unit": "uIU/mL" }
  ],

  "c_peptide": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "0.78-5.19", "unit": "ng/mL" }
  ],

  "homa_ir": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<2", "unit": "index" }
  ],

  "total_cholesterol": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<200", "unit": "mg/dL" }
  ],

  "ldl": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<100", "unit": "mg/dL" }
  ],

  "hdl": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": ">44", "unit": "mg/dL" }
  ],

  "cholesterol_hdl_ratio": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "2-5", "unit": "" }
  ],

  "triglycerides": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<150", "unit": "mg/dL" }
  ],

  "apo_a1": [
    { "gender": "male", "age_min": 18, "age_max": 120, "range": "37.1-72.1", "unit": "umol/L" },
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "38.6-80.3", "unit": "umol/L" }
  ],

  "apo_b": [
    { "gender": "male", "age_min": 18, "age_max": 120, "range": "80-155", "unit": "mg/dL" },
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "75-150", "unit": "mg/dL" }
  ],

  "apo_ratio": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "0.3-1.0", "unit": "" }
  ],

  "albumin": [
    { "gender": "all", "age_min": 18, "age_max": 60, "range": "3.5-5.2", "unit": "g/dL" },
    { "gender": "all", "age_min": 60, "age_max": 120, "range": "2.9-4.6", "unit": "g/dL" }
  ],

  "total_protein": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "6.6-8.8", "unit": "g/dL" }
  ],

  "alt": [
    { "gender": "male", "age_min": 0, "age_max": 120, "range": "10-50", "unit": "U/L" },
    { "gender": "female", "age_min": 0, "age_max": 120, "range": "10-35", "unit": "U/L" }
  ],

  "ast": [
    { "gender": "male", "age_min": 0, "age_max": 120, "range": "<50", "unit": "U/L" },
    { "gender": "female", "age_min": 0, "age_max": 120, "range": "<35", "unit": "U/L" }
  ],

  "alp": [
    { "gender": "male", "age_min": 19, "age_max": 120, "range": "40-129", "unit": "U/L" },
    { "gender": "female", "age_min": 17, "age_max": 120, "range": "35-104", "unit": "U/L" }
  ],

  "ggt": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "5-36", "unit": "U/L" }
  ],

  "ld": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "140-280", "unit": "U/L" }
  ],

  "globulin": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "2.5-3.5", "unit": "g/dL" }
  ],

  "albumin_globulin_ratio": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "0.9-2.2", "unit": "" }
  ],

  "magnesium": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "1.7-2.4", "unit": "mg/dL" }
  ],

  "total_bilirubin": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "0.1-1.1", "unit": "mg/dL" }
  ],

  "direct_bilirubin": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<0.6", "unit": "mg/dL" }
  ],

  "indirect_bilirubin": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "0.1-1.0", "unit": "mg/dL" }
  ],

  "ammonia": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "15-50", "unit": "µg/dL" }
  ],

  "hs_crp": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "<3", "unit": "mg/L" }
  ],

  "ck": [
    { "gender": "male", "age_min": 18, "age_max": 120, "range": "39-308", "unit": "U/L" },
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "26-192", "unit": "U/L" }
  ],

  "ck_mb": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "0-7.2", "unit": "ng/mL" }
  ],

  "homocysteine": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "<15", "unit": "umol/L" }
  ],

  "zinc": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "50-150", "unit": "ug/dL" }
  ],

  "copper": [
    { "gender": "male", "age_min": 18, "age_max": 120, "range": "64-140", "unit": "ug/dL" },
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "70-159", "unit": "ug/dL" }
  ],

  "selenium": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "460-1430", "unit": "ug/L" }
  ],

  "iron": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "65-175", "unit": "ug/dL" }
  ],

  "tibc": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "250-400", "unit": "ug/dL" }
  ],

  "transferrin": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "174-364", "unit": "mg/dL" }
  ],

  "vitamin_d": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "30-150", "unit": "ng/mL" }
  ],

  "vitamin_b12": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "187-883", "unit": "pg/mL" }
  ],

  "total_testosterone": [
    { "gender": "male", "age_min": 21, "age_max": 49, "range": "240-870", "unit": "ng/dL" },
    { "gender": "male", "age_min": 50, "age_max": 120, "range": "221-715", "unit": "ng/dL" },
    { "gender": "female", "age_min": 21, "age_max": 49, "range": "14-53", "unit": "ng/dL" },
    { "gender": "female", "age_min": 50, "age_max": 120, "range": "12-36", "unit": "ng/dL" }
  ],

  "free_testosterone": [
    { "gender": "male", "age_min": 18, "age_max": 120, "range": "8-34", "unit": "pg/mL" },
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "<4.4", "unit": "pg/mL" }
  ],

  "estrogen": [
    { "gender": "female", "age_min": 18, "age_max": 50, "range": "30-400", "unit": "pg/mL", "note": "varies with cycle" },
    { "gender": "female", "age_min": 50, "age_max": 120, "range": "0-30", "unit": "pg/mL", "note": "postmenopausal" }
  ],

  "progesterone": [
    { "gender": "female", "age_min": 18, "age_max": 50, "range": "0-1", "unit": "ng/mL", "note": "follicular phase" },
    { "gender": "female", "age_min": 18, "age_max": 50, "range": "5-20", "unit": "ng/mL", "note": "luteal phase" },
    { "gender": "female", "age_min": 50, "age_max": 120, "range": "0.09-0.78", "unit": "ng/mL", "note": "postmenopause" }
  ],

  "dhea_s": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "65.1-368", "unit": "ug/dL" }
  ],

  "shbg": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "11-78", "unit": "nmol/L", "note": "values <11 considered low, >78 high" }
  ],

  "lh": [
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "2.1-11.2", "unit": "mIU/mL", "note": "follicular phase" },
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "20-95", "unit": "mIU/mL", "note": "mid-cycle peak" },
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "1.2-11.2", "unit": "mIU/mL", "note": "luteal phase" },
    { "gender": "female", "age_min": 51, "age_max": 120, "range": "10-60", "unit": "mIU/mL", "note": "postmenopausal" },
    { "gender": "male", "age_min": 18, "age_max": 120, "range": "1.1-10", "unit": "mIU/mL" }
  ],

  "fsh": [
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "2.5-10.2", "unit": "mIU/mL", "note": "follicular phase" },
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "3.4-33.4", "unit": "mIU/mL", "note": "mid-cycle peak" },
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "1.5-9.1", "unit": "mIU/mL", "note": "luteal phase" },
    { "gender": "female", "age_min": 51, "age_max": 120, "range": "16-120", "unit": "mIU/mL", "note": "postmenopausal" },
    { "gender": "male", "age_min": 18, "age_max": 120, "range": "1-9.3", "unit": "mIU/mL" }
  ],

  "tsh": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "0.4-4.5", "unit": "uIU/mL", "note": "trimester-specific ranges exist" }
  ],

  "free_t3": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "2-4.2", "unit": "pg/mL" }
  ],

  "free_t4": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "0.9-1.75", "unit": "ng/dL" }
  ],

  "total_t3": [
    { "gender": "all", "age_min": 20, "age_max": 50, "range": "0.7-2.07", "unit": "ng/mL", "note": "adults 20-50" },
    { "gender": "all", "age_min": 50, "age_max": 90, "range": "0.35-1.93", "unit": "ng/mL", "note": "adults 50-90" }
  ],

  "total_t4": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "4.87-11.72", "unit": "ug/dL" }
  ],

  "reverse_t3": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "9-24", "unit": "ng/dL", "note": "commonly used reference (lab-specific)" }
  ],

  "tpo_ab": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "0-5.61", "unit": "IU/mL" }
  ],

  "tg_ab": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<115", "unit": "IU/mL" }
  ],

  "cortisol": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "5.9-19.9", "unit": "ug/dL", "note": "AM value typical; time-of-day matters" }
  ],

  "acth": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "7.2-63.3", "unit": "pg/mL" }
  ],

  "igf1": [
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "75-850", "unit": "ng/mL" },
    { "gender": "male", "age_min": 18, "age_max": 120, "range": "80-900", "unit": "ng/mL" }
  ],

  "leptin": [
    { "gender": "male", "age_min": 18, "age_max": 120, "range": "0.5-15", "unit": "ng/mL", "note": "wide variance by adiposity; females generally higher" },
    { "gender": "female", "age_min": 18, "age_max": 120, "range": "2-20", "unit": "ng/mL" }
  ],

  "adiponectin": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "2.5-6.0", "unit": "µg/mL", "note": "lab-dependent; values may be reported in mg/L" }
  ],

  "ca125": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "<35", "unit": "U/mL" }
  ],

  "ca15_3": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "<35", "unit": "U/mL" }
  ],

  "ca19_9": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "<37", "unit": "U/mL" }
  ],

  "psa": [
    { "gender": "male", "age_min": 18, "age_max": 120, "range": "0-4", "unit": "ng/mL" }
  ],

  "cea": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "<5", "unit": "ng/mL", "note": "smokers may have <10" }
  ],

  "calcitonin": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<10", "unit": "pg/mL", "note": "lab-dependent; many labs use <10 pg/mL as normal" }
  ],

  "afp": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<8.7", "unit": "ng/mL" }
  ],

  "tnf": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "0-8.1", "unit": "pg/mL", "note": "assay-dependent" }
  ],

  "ana": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<1:80", "unit": "titer", "note": "interpretation requires pattern and titre" }
  ],

  "ige": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "0-200", "unit": "IU/mL" }
  ],

  "igg": [
    { "gender": "all", "age_min": 18, "age_max": 120, "range": "7-16", "unit": "g/L", "note": "lab-dependent; pediatric ranges differ" }
  ],

  "anti_ccp": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<17", "unit": "U/mL" }
  ],

  "dsdna": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "0-6", "unit": "U/mL", "note": "7-12 borderline, >12 positive (lab dependent)" }
  ],

  "ssa_ssb": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<5", "unit": "U/mL", "note": "many labs differ; treat as qualitative" }
  ],

  "rnp": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<5", "unit": "AU/mL" }
  ],

  "sm_antibodies": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<0.5", "unit": "index", "note": "lab-dependent" }
  ],

  "anca": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<1:20", "unit": "titer", "note": "pattern (c-ANCA/p-ANCA) matters" }
  ],

  "anti_ena": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<0.5", "unit": "index", "note": "extractable nuclear antigen panel" }
  ],

  "il6": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "<7", "unit": "pg/mL", "note": "assay-dependent; elevated in inflammation" }
  ],

  "allergy_panel": [
    { "gender": "all", "age_min": 0, "age_max": 120, "range": "lab-specific", "unit": "", "note": "results vary widely by allergens tested; report as specific IgE values per allergen" }
  ]
})

def get_reference(biomarker_name: str, age: int, gender: str):
    entries = ref.get(biomarker_name, [])

    gfiltered = [e for e in entries if e["gender"] in [gender, "all"]]
    afinal = [e for e in gfiltered if e["age_min"] <= age <= e["age_max"]]

    return afinal if afinal else [e for e in entries if e["gender"] == "all"]

def build_prompt_context(user_input: BiomarkerRequest):
    data_dict = user_input.model_dump()

    age = data_dict["age"]
    gender = data_dict["gender"]

    result = {}

    for key, val in data_dict.items():
        if key in ["age", "gender", "height", "weight"]:
            continue
        if key not in ref:
            continue

        ranges = get_reference(key, age, gender)
        result[key] = {
            "value": val,
            "reference": ranges
        }

    return result




# ---------------- Cleaning Utility ----------------
def clean_json(data: Union[Dict, List, str]) -> Union[Dict, List, str]:
    """Recursively removes separators, extra whitespace, and artifacts from all string values."""
    if isinstance(data, str):
        text = re.sub(r"-{3,}", "", data)
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" -\n\t\r")
        return text
    elif isinstance(data, list):
        return [clean_json(i) for i in data if i and clean_json(i)]
    elif isinstance(data, dict):
        return {k.strip(): clean_json(v) for k, v in data.items()}
    return data


# ---------------- Parser ----------------
import re

def clean_line(t):
    return t.strip().replace("**", "").replace("###", "").strip()

def extract_status_and_explanation(block):
    status = ""
    explanation = ""

    status_match = re.search(r"Status:\s*(.*?)(?:\s*Explanation:|$)", block, re.S | re.I)
    exp_match = re.search(r"Explanation:\s*(.*)", block, re.S | re.I)

    if status_match:
        status = clean_line(status_match.group(1))
    if exp_match:
        explanation = clean_line(exp_match.group(1))

    return status, explanation


def parse_medical_report(text):

    data = {
        "executive_summary": {
            "top_health_priorities": [],
            "key_strengths": []
        },
        "system_analysis": {
            "kidney_function_test": {"status": "", "explanation": ""},
            "basic_checkup_cbc_hematology": {"status": "", "explanation": ""},
            "hormone_profile_comprehensive": {"status": "", "explanation": ""},
            "liver_function_test": {"status": "", "explanation": ""},
            "diabetic_profile": {"status": "", "explanation": ""},
            "lipid_profile": {"status": "", "explanation": ""},
            "cardiac_profile": {"status": "", "explanation": ""},
            "mineral_heavy_metal": {"status": "", "explanation": ""},
            "iron_profile": {"status": "", "explanation": ""},
            "bone_health": {"status": "", "explanation": ""},
            "vitamins": {"status": "", "explanation": ""},
            "thyroid_profile": {"status": "", "explanation": ""},
            "adrenal_stress_hormones": {"status": "", "explanation": ""},
            "blood_marker_cancer_profile": {"status": "", "explanation": ""},
            "immune_profile": {"status": "", "explanation": ""}
        },
        "personalized_action_plan": {
            "nutrition": "",
            "lifestyle": "",
            "testing": "",
            "medical_consultation": ""
        },
        "interaction_alerts": []
    }

    # ---------------------------
    # --- EXECUTIVE SUMMARY ----
    # ---------------------------
    exec_match = re.search(r"###\s*Executive Summary(.*?)(?=###|$)", text, re.S | re.I)
    if exec_match:
        block = exec_match.group(1)

        # Priorities (1., 2., 3.)
        priorities = re.findall(r"\d+\.\s*(.*)", block)
        data["executive_summary"]["top_health_priorities"] = [clean_line(p) for p in priorities]

        # Key strengths (- ...)
        strengths = re.findall(r"-\s*(.*)", block)
        data["executive_summary"]["key_strengths"] = [clean_line(s) for s in strengths]


    # ----------------------------
    # --- SYSTEM SPECIFIC BLOCK --
    # ----------------------------
    sys_match = re.search(r"###\s*System-Specific Analysis(.*?)(?=###|$)", text, re.S | re.I)
    if sys_match:
        sys_block = sys_match.group(1)

        # Map headings → your JSON keys
        heading_map = {
            r"Hormone Profile": "hormone_profile_comprehensive",
            r"Kidney Function Test": "kidney_function_test",
            r"Basic Check-up": "basic_checkup_cbc_hematology",
            r"Liver Function Test": "liver_function_test",
            r"Diabetic Profile": "diabetic_profile",
            r"Lipid Profile": "lipid_profile",
            r"Cardiac Profile": "cardiac_profile",
            r"Mineral & Heavy Metal": "mineral_heavy_metal",
            r"Iron Profile": "iron_profile",
            r"Bone Health": "bone_health",
            r"Vitamins": "vitamins",
            r"Thyroid Profile": "thyroid_profile",
            r"Adrenal Function": "adrenal_stress_hormones",
            r"Blood Marker Cancer Profile": "blood_marker_cancer_profile",
            r"Immune Profile": "immune_profile"
        }

        for pattern, key in heading_map.items():
            match = re.search(rf"\*\*{pattern}.*?\*\*(.*?)(?=\*\*|$)", sys_block, re.S | re.I)
            if match:
                block = match.group(1)
                status, explanation = extract_status_and_explanation(block)
                data["system_analysis"][key]["status"] = status
                data["system_analysis"][key]["explanation"] = explanation


    # ----------------------------
    # --- ACTION PLAN ------------
    # ----------------------------
    plan_match = re.search(r"###\s*Personalized Action Plan(.*?)(?=###|$)", text, re.S | re.I)
    if plan_match:
        plan_block = plan_match.group(1)

        for field in ["nutrition", "lifestyle", "testing", "medical_consultation"]:
            m = re.search(rf"\*\*{field.capitalize().replace('_',' ')}:\*\*\s*(.*)", plan_block, re.I)
            if m:
                data["personalized_action_plan"][field] = clean_line(m.group(1))


    # ----------------------------
    # --- INTERACTION ALERTS -----
    # ----------------------------
    alert_match = re.search(r"###\s*Interaction Alerts(.*)", text, re.S | re.I)
    if alert_match:
        a_block = alert_match.group(1)
        alerts = re.findall(r"-\s*(.*)", a_block)
        data["interaction_alerts"] = [clean_line(a) for a in alerts]

    return data


# ---------------- Endpoint ----------------
@app.post("/predict")
def predict(data: BiomarkerRequest):
    """Accepts biomarker input and returns structured and complete detailed medical insights."""
    # Build reference-enriched biomarker context
    try:
        # --- Prompt Template ---
        prompt = """
You are an advanced **Medical Insight Generation AI** trained to analyze **biomarkers and lab results**.
------------------------------
### Executive Summary
**Top  Health Priorities:**
1. ...
2. ...
3. ...
make it more detailed 

**Key Strengths:**
- ...
- ...
make it detailed
------------------------------
### System-Specific Analysis

**Kidney Function Test**
Status: Normal. Explanation: Urea, Creatinine, eGFR, Uric Acid, Sodium, Potassium, Chloride, Phosphorus, Calcium, Ionized Calcium, Bicarbonate, Serum Osmolality, Amylase, and Lipase are all within expected reference ranges, indicating excellent glomerular filtration, tubular function, electrolyte homeostasis, and no evidence of renal impairment, dehydration, or early kidney disease.

**Basic Check-up (CBC & Hematology)**
Status: Normal. Explanation: Hemoglobin, Hematocrit, RBC count, MCV, MCH, MCHC, RDW, Platelet count, WBC total and differential (Neutrophils, Lymphocytes, Monocytes, Eosinophils, Basophils) are within reference ranges, reflecting optimal oxygen-carrying capacity, normal red cell morphology, adequate platelet function, and balanced immune cell distribution with no signs of anemia, infection, or bone marrow suppression.

**Hormone Profile (Comprehensive)**
Status: Normal. Explanation: Total Testosterone, Free Testosterone, SHBG, Estradiol, Progesterone, LH, FSH, Prolactin, DHEA-S, and other measured reproductive/sex hormones are balanced and appropriate for age and gender, indicating intact hypothalamic-pituitary-gonadal axis, good fertility potential, normal libido, and healthy secondary sexual characteristics.

**Liver Function Test**
Status: Normal. Explanation: ALT, AST, ALP, GGT, LDH, Total Bilirubin, Direct & Indirect Bilirubin, Albumin, Globulin, Total Protein, Albumin/Globulin Ratio, and Ammonia are within reference ranges, demonstrating intact hepatocyte integrity, normal synthetic function, protein metabolism, and biliary excretion with no evidence of hepatic injury, cholestasis, cirrhosis, or metabolic liver disease.

**Diabetic Profile**
Status: Normal. Explanation: Fasting Blood Glucose, HbA1c, Fasting Insulin, C-Peptide, and HOMA-IR are all within optimal ranges, confirming excellent glycemic control, high insulin sensitivity, proper pancreatic beta-cell function, and very low risk of prediabetes or type 2 diabetes.

**Lipid Profile**
Status: Normal. Explanation: Total Cholesterol, LDL-C, HDL-C, Triglycerides, Non-HDL Cholesterol, Apo A-1, Apo B, Apo B/Apo A-1 Ratio, and Cholesterol/HDL Ratio are optimal, indicating low atherogenic risk, excellent cardiovascular protection, and minimal likelihood of plaque formation or coronary artery disease.

**Cardiac Profile**
Status: Normal. Explanation: hs-CRP, CK, CK-MB, Homocysteine, NT-proBNP (if measured), and other cardiac injury/inflammation markers are within normal limits, reflecting minimal systemic inflammation, healthy myocardial tissue, low thrombotic risk, and excellent long-term cardiovascular prognosis.

**Mineral & Heavy Metal**
Status: Normal. Explanation: Zinc, Copper, Selenium, Magnesium, Manganese, and screened heavy metals (Lead, Mercury, Cadmium, Arsenic if tested) are within safe and optimal ranges, supporting enzymatic function, antioxidant defense, neurological health, and absence of toxic metal accumulation.

**Iron Profile**
Status: Normal. Explanation: Serum Iron, TIBC, Transferrin Saturation, Ferritin, and Soluble Transferrin Receptor are balanced, indicating healthy iron stores, normal transport capacity, and no evidence of iron deficiency anemia, hemochromatosis, or chronic inflammation-related anemia.

**Bone Health**
Status: Normal. Explanation: Vitamin D (25-OH), Calcium, Phosphorus, Magnesium, Alkaline Phosphatase (bone isoform if available), PTH, and bone turnover markers (if tested) are optimal, supporting strong bone mineralization, healthy remodeling, and low risk of osteoporosis or osteomalacia.

**Vitamins**
Status: Normal. Explanation: Vitamin D (25-OH), Vitamin B12, Folate, Vitamin B6, Vitamin C, Vitamin A, Vitamin E, and Vitamin K (if measured) are within optimal ranges, ensuring robust immune function, neurological health, methylation, antioxidant protection, and prevention of deficiency-related disorders.

**Thyroid Profile**
Status: Normal. Explanation: TSH, Free T4, Free T3, Total T3, Total T4, Reverse T3, Anti-TPO Antibodies, and Anti-Thyroglobulin Antibodies are all within reference limits, confirming euthyroid status, normal hormone production and conversion, and absence of autoimmune thyroid disease.

**Adrenal Function / Stress Hormones / Other Hormones**
Status: Normal. Explanation: Morning Cortisol, ACTH, DHEA-S, IGF-1, Leptin, Adiponectin, Aldosterone (if tested), and Catecholamines/Metonephrines (if tested) are appropriately balanced, indicating resilient HPA axis, healthy stress response, growth hormone axis integrity, and optimal metabolic regulation.

**Blood Marker Cancer Profile**
Status: Normal. Explanation: CEA, CA19-9, CA125, CA15-3, AFP, PSA (men), HE4, ROMA score (if applicable), Calcitonin, and other tumor markers are within reference ranges, suggesting very low probability of active malignancy at this time (note: tumor markers are not screening tools and must be interpreted in clinical context).

**Immune Profile**
Status: Normal. Explanation: Immunoglobulin levels (IgG, IgA, IgM, IgE), ANA, ENA panel, Anti-dsDNA, Anti-CCP, ANCA, Complement C3/C4, IL-6, and lymphocyte subsets (if tested) are within normal limits, indicating competent humoral and cellular immunity with no evidence of immunodeficiency, active autoimmunity, or chronic inflammatory states.------------------------------

### Personalized Action Plan
**Nutrition:** ...
make it detailed
**Lifestyle:** ...
make it detailed
**Testing:** ...
make it detailed
**Medical Consultation:** ...
make it detailed
------------------------------
### Interaction Alerts
- ...
- ...
make it detailed
"""

        # --- Format User Data ---
        user_message = build_prompt_context(data)

        # --- Gemini Call ---
        model = genai.GenerativeModel(MODEL_ID)
        response = model.generate_content(f"{prompt}\n\n{user_message}")

        if not response or not getattr(response, "text", None):
            raise ValueError("Empty response from Gemini model.")

        report_text = response.text.strip()

        # --- Parse + Clean ---
        parsed_output = parse_medical_report(report_text)
        cleaned_output = clean_json(parsed_output)

        return cleaned_output

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")