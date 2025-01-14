from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
model_name = "OsBaran/Bert-Classification-Model-Tr-4"
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define pipeline
bert_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define input schema
class PredictionRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        prediction = bert_pipeline(request.text)
        return {"text": request.text, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")

@app.get("/")
async def root():
    return {"message": "BERT Classification API is running!"}
