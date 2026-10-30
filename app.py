from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np

# Model architecture re-define karna
class StockPredictionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockPredictionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 50)
        c0 = torch.zeros(2, x.size(0), 50)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

app = FastAPI(title="Stock Prediction API")

# Saved model aur scaler load karna
model = StockPredictionLSTM()
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()
scaler = joblib.load('scaler.pkl')

class InputData(BaseModel):
    prices: list[float] # Pichle 60 din ke prices

@app.post("/predict")
def predict_price(data: InputData):
    if len(data.prices) != 60:
        return {"error": "Please provide exactly 60 days of closing prices"}
    
    # Data preprocessing
    scaled_prices = scaler.transform(np.array(data.prices).reshape(-1, 1))
    input_tensor = torch.tensor(scaled_prices, dtype=torch.float32).unsqueeze(0)
    
    # Model prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        
    final_price = scaler.inverse_transform(prediction.numpy())[0][0]
    return {"predicted_next_day_price": float(final_price)}
