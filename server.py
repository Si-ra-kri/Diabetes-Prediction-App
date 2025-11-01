from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

# Load trained model
model = joblib.load("app/model.joblib")

# FastAPI app
app = FastAPI()

# Mount static & templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# ---------------- Home Page ----------------
@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


# ---------------- Prediction ----------------
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    pregnancies: float = Form(...),
    glucose: float = Form(...),
    blood_pressure: float = Form(...),
    skin_thickness: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    dpf: float = Form(...),  # diabetes pedigree function
    age: float = Form(...)
):
    # Prepare input features
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]])
    prediction = model.predict(features)[0]

    result = "Diabetic" if prediction == 1 else "Non-Diabetic"

    # Render result page
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "prediction": result}
    )
