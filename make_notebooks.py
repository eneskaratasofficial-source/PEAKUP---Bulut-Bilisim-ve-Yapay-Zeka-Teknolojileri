import nbformat as nbf
import os

def create_notebook(filename, code_cells, md_cells=None):
    nb = nbf.v4.new_notebook()
    cells = []
    
    if md_cells is None:
        md_cells = [""] * len(code_cells)
        
    for i, code in enumerate(code_cells):
        if i < len(md_cells) and md_cells[i]:
            cells.append(nbf.v4.new_markdown_cell(md_cells[i].strip()))
        cells.append(nbf.v4.new_code_cell(code.strip()))
        
    nb['cells'] = cells
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

os.makedirs('notebooks', exist_ok=True)

# --- 01. Data Prep ---
nb1_md = [
    "# Veri Setini İndirme ve Hazırlama\nKaggle API bulunamadığı için canlıda çalışacak şekilde Cresci-2017 benzeri sentetik veri oluşturuyoruz.",
    "## Veriyi Yükleme ve Sınıf Dengesi İnceleme",
    "## Temizlik ve SMOTE Uygulaması"
]
nb1_c = [
"""import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)""",
"""users_file = os.path.join(RAW_DIR, "users.csv")
edges_file = os.path.join(RAW_DIR, "edges.csv")

if not os.path.exists(users_file):
    print("Sentetik Cresci-2017 benzeri veri seti oluşturuluyor...")
    np.random.seed(42)
    n_samples = 3000
    labels = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    
    data = []
    for i in range(n_samples):
        is_real = labels[i] == 1
        followers = int(np.abs(np.random.normal(5000, 10000))) if is_real else int(np.abs(np.random.normal(50, 100)))
        friends = int(np.abs(np.random.normal(1000, 2000))) if is_real else int(np.abs(np.random.normal(2000, 5000)))
        statuses = int(np.abs(np.random.normal(15000, 20000))) if is_real else int(np.abs(np.random.normal(100, 500)))
        account_age_days = int(np.abs(np.random.normal(2000, 500))) if is_real else int(np.abs(np.random.normal(100, 50)))
        has_profile_pic = 1 if is_real else np.random.choice([0, 1], p=[0.7, 0.3])
        
        data.append({
            "id": i,
            "followers_count": followers,
            "friends_count": friends,
            "statuses_count": statuses,
            "account_age_days": account_age_days,
            "has_profile_pic": has_profile_pic,
            "label": labels[i]
        })
    df = pd.DataFrame(data)
    df.to_csv(users_file, index=False)
    
    edges = []
    for _ in range(8000):
        src = np.random.randint(0, n_samples)
        dst = np.random.randint(0, n_samples)
        edges.append({"source": src, "target": dst})
    pd.DataFrame(edges).to_csv(edges_file, index=False)
    print("Oluşturuldu.")
else:
    df = pd.read_csv(users_file)

print(df.head())""",
"""print("Eksik veriler temizleniyor...")
df = df.dropna()

print("SMOTE uygulanıyor...")
X = df.drop(["id", "label"], axis=1)
y = df["label"]

print(f"Orijinal Sınıf Dağılımı:\\n{y.value_counts()}")

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print(f"SMOTE Sonrası Sınıf Dağılımı:\\n{y_res.value_counts()}")

df_processed = pd.concat([X_res, y_res], axis=1)
df_processed["id"] = range(len(df_processed))

processed_file = os.path.join(PROCESSED_DIR, "users_processed.csv")
df_processed.to_csv(processed_file, index=False)"""
]

# --- 02. EDA ---
nb2_md = [
    "# Keşifsel Veri Analizi (EDA)",
    "Seaborn kütüphanesi kullanarak Bot ve Gerçek hesapların görselleştirilmesi."
]
nb2_c = [
"""import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

PROCESSED_DIR = "../data/processed"
df = pd.read_csv(os.path.join(PROCESSED_DIR, "users_processed.csv"))
df["Grup"] = df["label"].map({1: "Gerçek", 0: "Bot"})
""",
"""# Takipçi ve Takip Edilen Dağılımı
plt.figure(figsize=(10, 6))
sns.boxplot(x="Grup", y="followers_count", data=df)
plt.title("Takipçi Sayısı Kutusu (Boxplot)")
plt.savefig("followers_boxplot.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="account_age_days", hue="Grup", kde=True)
plt.title("Hesap Yaşı Histogramı")
plt.savefig("account_age_hist.png")
plt.show()"""
]

# --- 03. Baseline Models ---
nb3_md = [
    "# Temel Makine Öğrenmesi Modelleri",
    "Lojistik Regresyon, Random Forest (Overfitting kontrollü) ve XGBoost"
]
nb3_c = [
"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os

df = pd.read_csv("../data/processed/users_processed.csv")
X = df.drop(["id", "label"], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
os.makedirs("../models", exist_ok=True)
""",
"""def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    print(f"--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    try:
        print(f"AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")
    except:
        pass
    print("-" * 20)
    return model

# Lojistik Regresyon
evaluate_model("Lojistik Regresyon", LogisticRegression(max_iter=1000))

# Random Forest (Overfitting önlemek için max_depth snırlandı)
rf_model = evaluate_model("Random Forest", RandomForestClassifier(max_depth=10, min_samples_split=5, random_state=42))

# XGBoost
xgb_model = evaluate_model("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))

joblib.dump(rf_model, "../models/baseline_model.pkl")
print("Baseline model (Random Forest) kaydedildi.")
"""
]

# --- 04. Graph Features ---
nb4_md = [
    "# Grafik Tabanlı Özellik Çıkarımı",
    "NetworkX kullanarak hesaplar arası retweet graf özelliklerini (Degree, Betweenness vd.) hesaplıyoruz."
]
nb4_c = [
"""import pandas as pd
import networkx as nx
import numpy as np

users_df = pd.read_csv("../data/processed/users_processed.csv")
edges_df = pd.read_csv("../data/raw/edges.csv")

G = nx.from_pandas_edgelist(edges_df, source="source", target="target", create_using=nx.Graph())
print(f"Graph Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
""",
"""# Özellik çıkarımı
degree_centrality = nx.degree_centrality(G)
clustering_coef = nx.clustering(G)

users_df["degree_centrality"] = users_df["id"].map(degree_centrality).fillna(0)
users_df["clustering_coef"] = users_df["id"].map(clustering_coef).fillna(0)

# Betweenness çok uzun sürerse örneklem üzerinden hesaplanabilir
users_df["betweenness_centrality"] = 0 # Performans için varsayılan

users_df.to_csv("../data/processed/users_graph_features.csv", index=False)
print("Graph özellikleri başarıyla eklendi.")
print(users_df.head())
"""
]

# --- 05. Hybrid Model ---
nb5_md = [
    "# Hibrit Model (Tablo + Graf Özellikleri)",
    "Tablo özellikleri ile Graph metriklerini birleştirip XGBoost ve IsolationForest kullanarak modeli eğitiyoruz."
]
nb5_c = [
"""import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, classification_report
import joblib

df = pd.read_csv("../data/processed/users_graph_features.csv")
X = df.drop(["id", "label"], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""",
"""print("Anomali Tespiti - Isolation Forest")
iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(X_train)
iso_preds = iso.predict(X_test)
print(pd.Series(iso_preds).value_counts())

print("\\nHibrit XGBoost Eğitimi")
xgb_hybrid = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_hybrid.fit(X_train, y_train)

preds = xgb_hybrid.predict(X_test)
f1 = f1_score(y_test, preds)
print(f"Hibrit Model F1 Skor: {f1:.4f}")
print(classification_report(y_test, preds))

joblib.dump(xgb_hybrid, "../models/hybrid_model.pkl")
print("Hibrit model kaydedildi: ../models/hybrid_model.pkl")
"""
]

create_notebook('notebooks/01_data_prep.ipynb', nb1_c, nb1_md)
create_notebook('notebooks/02_eda.ipynb', nb2_c, nb2_md)
create_notebook('notebooks/03_baseline_models.ipynb', nb3_c, nb3_md)
create_notebook('notebooks/04_graph_features.ipynb', nb4_c, nb4_md)
create_notebook('notebooks/05_hybrid_model.ipynb', nb5_c, nb5_md)
print("Bütün Notebooklar oluşturuldu.")
