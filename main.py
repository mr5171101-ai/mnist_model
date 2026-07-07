# main.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
from PIL import Image

import numpy as np
from model_handler import load_mnist_model, preprocess_image

app = FastAPI(title="MNIST Digit Classification API")

# Model ko startup par load karna
model = load_mnist_model()

@app.get("/")
def home():
    return {"message": "MNIST ANN API is running!"}

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    # Uploaded file ko read karke image mein convert karna
    request_object_content = await file.read()
    image = Image.open(io.BytesIO(request_object_content))
    
    # Image preprocessing
    processed_image = preprocess_image(image)
    
    # Model Prediction
    predictions = model.predict(processed_image)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    
    return {
        "prediction": predicted_class,
        "confidence": f"{confidence * 100:.2f}%"
    }

if __name__ == "__main__":
    # Yahan tabdeeli ki hai: 'main' ko hata kar 'app' ya "main:app" likha hai
    uvicorn.run("main:app", host="127.0.0.1", port=8000)