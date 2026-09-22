from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestRegressor

# Customized Transformer (required to load .pkl)
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        X_new = X.copy()
        if 'gender' in X_new.columns:
            X_new['gender'] = X_new['gender'].map({'Male': 1, 'Female': 0})
        if 'extracurricular_activities' in X_new.columns:
            X_new['extracurricular_activities'] = X_new['extracurricular_activities'].map({'Yes': 1, 'No': 0})
            
        X_new['total_experience'] = (X_new['internship_count'] + X_new['live_projects'] + (X_new['work_experience_months'] / 12))
        X_new['skill_score_combined'] = (X_new['technical_skill_score'] + X_new['soft_skill_score']) / 2
        X_new['academic_score_avg'] = (X_new['ssc_percentage'] + X_new['hsc_percentage'] + X_new['degree_percentage'] + X_new['cgpa'] * 10) / 4
        X_new['engagement_score'] = (X_new['attendance_percentage'] + X_new['extracurricular_activities'])
        
        X_new.fillna(0, inplace=True)
        return X_new

# Initialized App and Load Model
app = FastAPI(title="Placement & Salary Prediction API")

clf_model = joblib.load("artifacts/best_clf_pipeline.pkl")
reg_model = joblib.load("artifacts/best_reg_pipeline.pkl")

# Pydantic Schema 
class CandidateInput(BaseModel):
    gender: str
    ssc_percentage: float
    hsc_percentage: float
    degree_percentage: float
    cgpa: float
    entrance_exam_score: float
    backlogs: int
    internship_count: int
    live_projects: int
    work_experience_months: int
    technical_skill_score: float
    soft_skill_score: float
    certifications: int
    attendance_percentage: float
    extracurricular_activities: str

COLUMNS_ORDER = [
    'gender', 'ssc_percentage', 'hsc_percentage', 'degree_percentage', 'cgpa','entrance_exam_score', 'technical_skill_score', 'soft_skill_score',''
    'internship_count', 'live_projects', 'work_experience_months','certifications', 'attendance_percentage', 'backlogs', 'extracurricular_activities'
]

# Predict
@app.get("/")
def read_root():
    return {"message": "Placement Prediction API is running."}

@app.post("/predict")
def predict(input_data: CandidateInput):
    try: 
        df = pd.DataFrame([input_data.model_dump()])[COLUMNS_ORDER]
        numeric_cols = df.select_dtypes(include=['int64', 'int32']).columns
        df[numeric_cols] = df[numeric_cols].astype(float)

        # Classification
        placement_pred = int(clf_model.predict(df)[0])
        placement_prob = float(clf_model.predict_proba(df)[0][1])

        # Regression
        salary_pred = 0.0
        if placement_pred == 1:
            salary_pred = max(0.0, float(reg_model.predict(df)[0]))

        return {
            "placement_status": "Placed" if placement_pred == 1 else "Not Placed",
            "placement_probability": placement_prob,
            "estimated_salary_lpa": salary_pred
        }

    except Exception as e:
        return {"error": str(e)}