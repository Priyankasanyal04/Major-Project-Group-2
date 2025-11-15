import scipy.io
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------
# 1. Load EEG data from A07.mat
# ---------------------------
mat_path = r"C:\Users\KIIT\Desktop\major_project _2\A07.mat" 
mat = scipy.io.loadmat(mat_path)
data_struct = mat["data"][0, 0]

# EEG matrix (347704 x 8)
eeg = np.array(data_struct[1])
# Labels (347704 x 1)
labels = np.array(data_struct[2]).flatten()

print("EEG shape:", eeg.shape, " Labels shape:", labels.shape)

# ---------------------------
# 2. Preprocess: normalize + segment
# ---------------------------
# Scale each channel
scaler = StandardScaler()
eeg = scaler.fit_transform(eeg)

# Segment length (e.g. 250 samples ≈ 1 s at 250 Hz)
window = 250
stride = 125  # 50% overlap

segments = []
segment_labels = []
for start in range(0, len(eeg) - window, stride):
    end = start + window
    X_seg = eeg[start:end]
    y_seg = int(np.round(np.mean(labels[start:end])))  # majority label
    segments.append(X_seg)
    segment_labels.append(y_seg)

X = np.stack(segments)          # shape (num_segments, window, channels)
y = np.array(segment_labels)    # shape (num_segments,)
print("Segmented:", X.shape, y.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------
# 3. Dataset class
# ---------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_ds = EEGDataset(X_train, y_train)
test_ds = EEGDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# ---------------------------
# 4. CNN-LSTM model
# ---------------------------
class EEGNet(nn.Module):
    def __init__(self, n_channels=8, hidden=64, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (batch, time, channels)
        x = x.permute(0, 2, 1)            # -> (batch, channels, time)
        x = self.cnn(x)                   # -> (batch, 64, time//4)
        x = x.permute(0, 2, 1)            # -> (batch, time', 64)
        _, (h, _) = self.lstm(x)          # h: (1, batch, hidden)
        out = self.fc(h[-1])              # -> (batch, num_classes)
        return out

# ---------------------------
# 5. Training loop
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGNet().to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0
    for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        Xb, yb = Xb.to(device), yb.to(device)
        optim.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Train loss: {total_loss/len(train_loader):.4f}")

# ---------------------------
# 6. Evaluation
# ---------------------------
# ---------------------------
# 6. Evaluation + Plots
# ---------------------------
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        preds = logits.argmax(dim=1)
        y_true.extend(yb.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = (y_true == y_pred).mean()
print(f"✅ Test Accuracy: {acc:.3f}")

# ---- Confusion matrix (values + heatmap) ----
labels_unique = np.unique(np.concatenate([y_true, y_pred]))
cm = confusion_matrix(y_true, y_pred, labels=labels_unique)
print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)
print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=3))

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', savepath='confusion_matrix.png'):
    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm = cm.astype('float') / np.maximum(cm_sum, 1)
    plt.figure(figsize=(5,4), dpi=140)
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=9)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

class_names = [f"Class {c}" for c in labels_unique]  # customize names if you like
plot_confusion_matrix(cm, class_names, normalize=True, title='Confusion Matrix (normalized)', savepath='confusion_matrix_norm.png')
plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion Matrix (counts)', savepath='confusion_matrix_counts.png')
print("🖼️ Saved: confusion_matrix_norm.png, confusion_matrix_counts.png")

# ---- EEG heatmaps: class-wise average window (time x channels) ----
# Build per-class averaged window to visualize typical pattern
# We’ll reuse the exact segmentation scheme used for training.
def build_segments(eeg_array, labels_array, window=250, stride=125):
    segs, labs = [], []
    for start in range(0, len(eeg_array) - window, stride):
        end = start + window
        X_seg = eeg_array[start:end]               # (window, channels)
        y_seg = int(np.round(np.mean(labels_array[start:end])))
        segs.append(X_seg)
        labs.append(y_seg)
    return np.stack(segs), np.array(labs)

# NOTE: use the same 'eeg' and 'labels' and same StandardScaler as training
# If 'eeg' was overwritten, rebuild it above exactly like in training.
X_all, y_all = build_segments(eeg, labels, window=250, stride=125)

def class_average_window(X, y, cls):
    if np.sum(y == cls) == 0:
        return None
    return X[y == cls].mean(axis=0)  # (window, channels)

avg_non = class_average_window(X_all, y_all, 0)
avg_tar = class_average_window(X_all, y_all, 1)

def plot_eeg_heatmap(avg_win, title, savepath):
    if avg_win is None:
        print(f"⚠️  No samples for {title}, skipping plot.")
        return
    # avg_win shape: (time, channels)
    plt.figure(figsize=(6,3), dpi=140)
    plt.imshow(avg_win.T, aspect='auto', origin='lower')
    plt.title(title)
    plt.ylabel('Channels')
    plt.xlabel('Time (samples)')
    plt.colorbar(label='Amplitude (z-scored)')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    print(f"🖼️ Saved: {savepath}")

plot_eeg_heatmap(avg_non, "Average EEG Window — NonTarget", "eeg_heatmap_nontarget.png")
plot_eeg_heatmap(avg_tar, "Average EEG Window — Target", "eeg_heatmap_target.png")

# Save the trained model (as before)
torch.save(model.state_dict(), "A07_EEG_classifier.pth")
