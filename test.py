import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# --- 2. Load Dataset ---
# Upload your Kaggle dataset manually or mount Google Drive
# For demo purposes, I'll generate a small mock dataset:

# Replace this with actual loading if you have the real Kaggle CSV
df = pd.read_csv('data/Combined Data.csv')

# Replace this with actual loading if you have the real Kaggle CSV
# Example:
# df = pd.read_csv('/path/to/mental_health_sentiment.csv')
# --- 2. Load and Clean Dataset ---

# 删除 statement 或 label 中任何为空的行
df = df.dropna(subset=['statement', 'status'])

# 确保 statement 都是字符串类型
df['statement'] = df['statement'].astype(str)

# 映射标签
label_map = {label: idx for idx, label in enumerate(df['status'].unique())}
reverse_label_map = {idx: label for label, idx in label_map.items()}

# 准备文本和标签
texts = df['statement'].tolist()
y = df['status'].map(label_map).tolist()

# --- 3. Prepare Embeddings ---
print("\nLoading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("\nEncoding text to embeddings...")
X = model.encode(texts)

print(df.head())




# --- 4. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Train Classifier ---
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
import numpy as np

# --- 6. Evaluate ---
y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_map.keys()))

# 基本指标
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Accuracy: {acc:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"Weighted F1 Score: {f1:.4f}")

# 如果是二分类，还可以尝试计算 ROC AUC（需将 y_test 和 y_pred 转为二值）
if len(set(y)) == 2:
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"ROC AUC Score: {auc:.4f}")

# --- 7. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
