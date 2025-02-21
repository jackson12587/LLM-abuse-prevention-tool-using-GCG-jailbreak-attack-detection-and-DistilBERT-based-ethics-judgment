import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import joblib  # ✅ 用于保存 StandardScaler
import os

# =========================
# **1. 设置 GPU 设备**
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 训练设备: {device}")

# =========================
# **2. 读取数据**
# =========================
train_file = r"your file"  
test_file = r"your file"    

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# **打乱训练集数据，保持测试集顺序**
train_df = shuffle(train_df, random_state=42).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# =========================
# **3. 数据预处理**
# =========================
X_train = train_df[['LNR', 'ABF', 'P']].values
y_train = train_df['label'].values  

X_test = test_df[['LNR', 'ABF', 'P']].values
y_test = test_df['label'].values  

# ✅ **修正归一化问题：训练时保存 StandardScaler**
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# **保存 StandardScaler**
scaler_path = r"your file"
joblib.dump(scaler, scaler_path)
print(f"✅ StandardScaler 已保存至: {scaler_path}")

# ✅ **转换为 PyTorch 张量，并移动到 GPU**
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# =========================
# **4. 定义 MLP 模型**
# =========================
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 32)  
        self.bn1 = nn.BatchNorm1d(32)  # ✅ **BatchNorm1d 防止数值过大**
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)  
        self.bn2 = nn.BatchNorm1d(16)  # ✅ **BatchNorm1d**
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 2)   
        self.dropout = nn.Dropout(0.1)  # ✅ **减少 Dropout 避免训练和推理分布不匹配**

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# **初始化模型，并移动到 GPU**
model = MLPClassifier().to(device)

# =========================
# **5. 训练模型**
# =========================
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # ✅ **降低学习率**

# 训练超参数
batch_size = 32  
epochs = 400  

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# **训练循环**
losses = []
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    losses.append(loss.item())
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# =========================
# **6. 评估模型**
# =========================
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = torch.argmax(y_pred, dim=1)
    accuracy = (y_pred_class == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f"✅ 测试集分类准确率: {accuracy:.4f}")

# =========================
# **7. 保存模型**
# =========================
model_path = r"your file"
torch.save(model.state_dict(), model_path)
print(f"✅ 训练完成的模型已保存至: {model_path}")

# =========================
# **8. 保存预测结果**
# =========================
test_df['Predicted Label'] = y_pred_class.cpu().numpy()
output_pred_path = r"your file"
test_df.to_csv(output_pred_path, index=False)
print(f"✅ 预测结果已保存至: {output_pred_path}")

# =========================
# **9. 训练损失曲线**
# =========================
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training Loss Curve")
plt.show()
