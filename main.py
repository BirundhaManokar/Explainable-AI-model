"""
main.py — BankIQ Backend
Run: python main.py
API Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn, json, os

from claude_service import ask_claude, shap_to_plain, simplify
from ml_models import predict_loan, predict_churn

app = FastAPI(title="BankIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ══════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════

class Msg(BaseModel):
    role: str
    content: str

class AskReq(BaseModel):
    question: str
    profile: str = "Student"
    language: str = "English"
    topic: str = "Loans & EMI"
    history: List[Msg] = []

class SimplifyReq(BaseModel):
    previous_answer: str
    profile: str = "Student"
    language: str = "English"

class LoanReq(BaseModel):
    income: float
    loan_amount: float
    cibil_score: int = 700
    education: str = "Graduate"
    self_employed: str = "No"
    loan_term: int = 12
    dependents: int = 0
    residential_assets: float = 0
    commercial_assets: float = 0
    luxury_assets: float = 0
    bank_assets: float = 0
    profile: str = "Student"
    language: str = "English"

class ChurnReq(BaseModel):
    tenure: int
    balance: float
    products: int = 1
    is_active: bool = True
    complaints: int = 0
    credit_score: int = 650
    age: int = 35
    salary: float = 50000
    satisfaction: int = 3
    points: int = 400
    profile: str = "Student"
    language: str = "English"

class FeedbackReq(BaseModel):
    question: str
    answer: str
    rating: int
    profile: str
    language: str
    comment: str = ""

# ══════════════════════════════════════
# ROUTES
# ══════════════════════════════════════

@app.get("/")
def root():
    return {
        "status": "BankIQ API Running!",
        "docs": "http://localhost:8000/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "loan_model":  "Ready" if os.path.exists("models/loan_model.pkl")  else "Run ml_models.py",
        "churn_model": "Ready" if os.path.exists("models/churn_model.pkl") else "Run ml_models.py"
    }

@app.post("/api/ask")
async def ask(req: AskReq):
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        data = await ask_claude(req.question, req.profile, req.language, req.topic, history)
        return {"success": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simplify")
async def simplify_api(req: SimplifyReq):
    try:
        data = await simplify(req.previous_answer, req.profile, req.language)
        return {"success": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/loan/predict")
async def loan_api(req: LoanReq):
    try:
        approved, confidence, shap_dict = predict_loan(
            income=req.income, loan_amount=req.loan_amount,
            cibil_score=req.cibil_score, education=req.education,
            self_employed=req.self_employed, loan_term=req.loan_term,
            dependents=req.dependents, residential_assets=req.residential_assets,
            commercial_assets=req.commercial_assets, luxury_assets=req.luxury_assets,
            bank_assets=req.bank_assets
        )
        explanation = await shap_to_plain(shap_dict, approved, req.profile, req.language, "loan")
        return {
            "success": True,
            "prediction": "Approved" if approved else "Rejected",
            "confidence": confidence,
            "shap": shap_dict,
            "explanation": explanation
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Loan model not found! Run: python ml_models.py")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/churn/predict")
async def churn_api(req: ChurnReq):
    try:
        will_churn, risk_score, shap_dict = predict_churn(
            tenure=req.tenure, balance=req.balance, products=req.products,
            is_active=req.is_active, complaints=req.complaints,
            credit_score=req.credit_score, age=req.age,
            salary=req.salary, satisfaction=req.satisfaction, points=req.points
        )

        # Rule-based explanation — no Groq call to avoid JSON parsing errors
        if will_churn:
            if req.profile == "Student":
                pts = [
                    f"Customer complained {req.complaints} times — like failing multiple subjects",
                    f"Low balance Rs.{req.balance} — not actively using the bank",
                    f"Satisfaction score {req.satisfaction}/5 — very unhappy with service"
                ]
                analogy = "Like a student who stopped attending class — likely to drop out!"
            elif req.profile == "Professional":
                pts = [
                    f"Complaint count: {req.complaints} — high dissatisfaction index",
                    f"Balance: Rs.{req.balance} — low engagement with bank products",
                    f"Satisfaction: {req.satisfaction}/5 — below retention threshold"
                ]
                analogy = "Customer NPS critically low — immediate retention action required."
            else:
                pts = [
                    f"Customer complained {req.complaints} times — not happy with bank",
                    f"Money in account Rs.{req.balance} is low — not using bank much",
                    f"Happiness score {req.satisfaction}/5 — needs urgent care"
                ]
                analogy = "Like a regular at a tea shop who stopped coming — something upset them!"
        else:
            if req.profile == "Student":
                pts = [
                    f"With bank for {req.tenure} years — loyal like a dedicated student",
                    f"Uses {req.products} products — actively engaged",
                    f"Satisfaction score {req.satisfaction}/5 — happy with service"
                ]
                analogy = "Like a student who loves their school — not going anywhere!"
            elif req.profile == "Professional":
                pts = [
                    f"Tenure: {req.tenure} years — strong retention indicator",
                    f"Product usage: {req.products} — good cross-sell engagement",
                    f"Satisfaction: {req.satisfaction}/5 — above average"
                ]
                analogy = "High CLV customer with strong retention probability."
            else:
                pts = [
                    f"With bank for {req.tenure} years — very loyal customer",
                    f"Uses {req.products} bank services — happy and engaged",
                    f"Satisfaction score {req.satisfaction}/5 — good experience"
                ]
                analogy = "Like a loyal customer at your favourite shop — comes back every time!"

        explanation = {
            "explanation_points": pts,
            "analogy": analogy,
            "transparency_score": 82
        }

        return {
            "success": True,
            "will_churn": will_churn,
            "risk_score": risk_score,
            "verdict": "High Churn Risk" if will_churn else "Customer Will Stay",
            "shap": shap_dict,
            "explanation": explanation
        }

    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Churn model not found! Run: python ml_models.py")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def feedback_api(req: FeedbackReq):
    try:
        data = []
        if os.path.exists("feedback.json"):
            with open("feedback.json", "r") as f:
                data = json.load(f)
        data.append({
            "question": req.question, "answer": req.answer,
            "rating": req.rating, "profile": req.profile,
            "language": req.language, "comment": req.comment
        })
        with open("feedback.json", "w") as f:
            json.dump(data, f, indent=2)
        return {"success": True, "message": "Feedback saved!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feedback/stats")
def feedback_stats():
    try:
        if not os.path.exists("feedback.json"):
            return {"total": 0, "helpful": 0, "not_helpful": 0}
        with open("feedback.json", "r") as f:
            data = json.load(f)
        helpful = sum(1 for d in data if d["rating"] == 1)
        return {
            "total": len(data), "helpful": helpful,
            "not_helpful": len(data) - helpful,
            "rate": f"{helpful/len(data)*100:.1f}%" if data else "N/A"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("=" * 50)
    print("BankIQ Backend Starting...")
    print("API Docs -> http://localhost:8000/docs")
    print("Health   -> http://localhost:8000/health")
    print("=" * 50)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
