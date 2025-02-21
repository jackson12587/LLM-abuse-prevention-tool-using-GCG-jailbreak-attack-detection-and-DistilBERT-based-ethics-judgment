import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib  # ✅ 用于加载 StandardScaler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import re
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree
import time
import os

# ✅ **设置 MLP 模型和 StandardScaler 归一化路径**
mlp_model_path = r"your file"
scaler_path = r"your file"

# ✅ **初始化 StanfordCoreNLP**
nlp_path = r'your file'
nlp = StanfordCoreNLP(nlp_path)

# ✅ **初始化 GPT-2 语言模型**
tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt = GPT2LMHeadModel.from_pretrained('gpt2')
model_gpt.eval()

# ✅ **加载 DistilBERT 有害内容分类器**
classifier_distilbert = pipeline("text-classification", 
                                 model=r"your file", 
                                 tokenizer=r"your file",
                                 device=0)

# ✅ **加载 StandardScaler**
scaler = joblib.load(scaler_path)
print(f"✅ 加载 StandardScaler 归一化参数: {scaler_path}")

# ✅ **定义 MLP 分类模型**
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 2)   
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# ✅ **安全加载 MLP 模型**
mlp_model = MLP()
mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=torch.device('cpu')))  # ✅ 显式指定设备
mlp_model.eval()
print(f"✅ MLP 模型已加载: {mlp_model_path}")

# ✅ **函数：解析句法树，计算特征**
def analyze_tree(tree):
    depth = tree.height() - 1
    total_nodes = sum(1 for _ in tree.subtrees())
    leaves = len(tree.leaves())
    non_leaves = total_nodes - leaves
    average_branching_factor = sum(len(subtree) for subtree in tree.subtrees(lambda t: t.height() > 2)) / non_leaves if non_leaves > 0 else 0
    lnr = leaves / total_nodes
    return depth, total_nodes, leaves, average_branching_factor, lnr

# ✅ **函数：计算 Perplexity**
def calculate_perplexity(sentence, model, tokenizer):
    encodings = tokenizer(sentence, return_tensors='pt')
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

# ✅ **GCG 分类函数**
def classify_text(text):
    parse_result = nlp.parse(text)
    tree = Tree.fromstring(parse_result)
    depth, total_nodes, leaves, average_branching_factor, lnr = analyze_tree(tree)
    perplexity = calculate_perplexity(text, model_gpt, tokenizer_gpt)

    # **归一化输入数据**
    input_data = np.array([[lnr, average_branching_factor, perplexity]])
    input_data = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # **MLP 推理**
    with torch.no_grad():
        output = mlp_model(input_tensor)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    # **DistilBERT 进行有害内容分类**
    bert_result = classifier_distilbert([text])[0]
    bert_label = "harmful" if bert_result['label'] == 'LABEL_1' else "safe"
    bert_confidence = bert_result['score']

    # **🧐 输出调试信息到 GUI**
    result_text = (
        f"🧐 句子: {text}\n"
        f"📊 特征: lnr={lnr:.4f}, abf={average_branching_factor:.4f}, p={perplexity:.2f}\n"
        f"🔍 Logits: {output.numpy()}\n"
        f"🔹 softmax 计算后: safe={probs[0][0]:.4f}, gcg={probs[0][1]:.4f}\n"
        f"🛑 GCG 预测类别: {'safe' if predicted_class == 0 else 'gcg'}\n"
        f"⚠️ DistilBERT 预测: {bert_label}（置信度: {bert_confidence:.4f}）\n"
        f"------------------------------\n"
    )
    
    output_textbox.insert(tk.END, result_text)

    return "safe" if predicted_class == 0 else "gcg", bert_label, bert_confidence

# ✅ **单个文本处理**
def process_text():
    input_text = input_textbox.get("1.0", tk.END).strip()
    if not input_text:
        return

    output_textbox.delete("1.0", tk.END)
    
    for sentence in input_text.split("\n"):
        classify_text(sentence)

# ✅ **批量处理文件**
def process_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
    if not file_path:
        return

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            messagebox.showerror("错误", "只支持 CSV 或 Excel 文件")
            return
    except Exception as e:
        messagebox.showerror("文件读取错误", str(e))
        return

    if 'sentence' not in df.columns:
        messagebox.showerror("错误", "文件中必须包含'sentence'列")
        return

    # **批量分类**
    results = []
    for sentence in df['sentence']:
        gcg_result, bert_label, bert_confidence = classify_text(sentence)
        results.append([gcg_result, bert_label, round(bert_confidence, 4)])

    # **添加结果列**
    df['GCG 分类'] = [r[0] for r in results]
    df['DistilBERT 分类'] = [r[1] for r in results]
    df['DistilBERT 置信度'] = [r[2] for r in results]

    # **保存结果**
    output_dir = r"your file"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(file_path))

    if file_path.endswith('.csv'):
        df.to_csv(output_path, index=False, encoding='utf-8')
    elif file_path.endswith('.xlsx'):
        df.to_excel(output_path, index=False)

    messagebox.showinfo("完成", f"处理完成，文件已保存至: {output_path}")

# ✅ **创建 GUI**
root = tk.Tk()
root.title("文本分类")
root.geometry("800x600")

input_textbox = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=10)
input_textbox.grid(row=0, column=0, padx=10, pady=10)

output_textbox = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=15)
output_textbox.grid(row=1, column=0, padx=10, pady=10)

tk.Button(root, text="分类", command=process_text).grid(row=2, column=0, padx=10, pady=10)
tk.Button(root, text="上传文件并分类", command=process_file).grid(row=3, column=0, padx=10, pady=10)

root.mainloop()
