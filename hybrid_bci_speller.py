import torch
import numpy as np
from scipy.io import loadmat
from predictive_lstm import PredictiveLSTM, predict_top_k
from eeg_classifier import EEGNet
from tokenizers import Tokenizer

# --------------------------------------------------
# 1️⃣ Load EEG classifier
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eeg_model = EEGNet()
eeg_model.load_state_dict(torch.load("A07_EEG_classifier.pth", map_location=device))
eeg_model.to(device)
eeg_model.eval()

# --------------------------------------------------
# 2️⃣ Load Predictive LSTM model
# --------------------------------------------------
checkpoint = torch.load("predictive_lstm.pth", map_location=device)
tokenizer = Tokenizer.from_str(checkpoint["tokenizer"])
vocab_size = tokenizer.get_vocab_size()
sd = checkpoint["state_dict"]

# infer dims from weights
emb_dim = sd["lstm.weight_ih_l0"].shape[1]      # typically 128
hidden_dim = sd["proj.bias"].shape[0]           # typically 192
pad_idx = tokenizer.token_to_id("[PAD]") if "[PAD]" in tokenizer.get_vocab() else 0

lm_model = PredictiveLSTM(
    vocab_size=vocab_size,
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    num_layers=2,
    pad_idx=pad_idx
).to(device)

lm_model.load_state_dict(sd)
lm_model.eval()


# --------------------------------------------------
# 3️⃣ Load sample EEG data (simulate real-time input)
# --------------------------------------------------
mat = loadmat(r"C:\Users\KIIT\Desktop\major_project _2\A07.mat")
data = mat["data"][0, 0]
eeg_matrix = np.array(data[1])  # EEG (samples × 8)
labels = np.array(data[2]).flatten()

# Normalization (same as during training)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
eeg_matrix = scaler.fit_transform(eeg_matrix)

# --------------------------------------------------
# 4️⃣ Simulate EEG-based symbol detection
# --------------------------------------------------
# Take one EEG segment (just for demo)
segment = eeg_matrix[0:250, :]   # 1 second of data
segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    pred_class = torch.argmax(eeg_model(segment_tensor), dim=1).item()

# Map EEG prediction (0/1) to a symbol (customize for your layout)
symbol_map = {0: "A", 1: "H"}  # Example mapping for demo
predicted_symbol = symbol_map.get(pred_class, "?")
print(f"🧠 EEG predicted symbol: {predicted_symbol}")

# --------------------------------------------------
# 5️⃣ Feed predicted symbol into Predictive LSTM
# --------------------------------------------------
context_text = predicted_symbol  # can also use entire typed text
predictions = predict_top_k(lm_model, tokenizer, context_text, k=5, device=device)

print("\n🔤 Predicted next letters/words:")
for word, prob in predictions:
    print(f"  {word}  ({prob:.3f})")

# --------------------------------------------------
# 6️⃣ (Optional) Loop simulation for real-time use
# --------------------------------------------------
# typed_text = ""
# while True:
#     symbol = get_symbol_from_EEG()     # from EEG model
#     typed_text += symbol
#     preds = predict_top_k(lm_model, tokenizer, typed_text, k=5, device=device)
#     update_UI(preds)
