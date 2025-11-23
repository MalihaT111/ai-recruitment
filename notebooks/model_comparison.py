"""
Model Comparison Framework for Resume-Job Matching
Compares multiple ML models and embedding approaches
"""

import numpy as np
import pandas as pd
import joblib
import time
import math
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================
# 1. LOAD DATA
# ============================================
print("Loading data...")
train_df = pd.read_csv("../data/train_pairs_rich.csv")
test_df = pd.read_csv("../data/test_pairs_rich.csv")

FEATURES = [
    "keyword_overlap",
    "skill_score",
    "experience_score",
    "education_score",
    "domain_match",
]

X_train = train_df[FEATURES]
y_train = train_df["label"]
X_test = test_df[FEATURES]
y_test = test_df["label"]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Class balance - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")


# ============================================
# 2. DEFINE MODELS TO COMPARE
# ============================================
models = {
    # "Logistic Regression": LogisticRegression(
    #     max_iter=1000,
    #     random_state=42,
    #     class_weight='balanced'
    # ),
    
    # "Random Forest": RandomForestClassifier(
    #     n_estimators=100,
    #     max_depth=10,
    #     random_state=42,
    #     class_weight='balanced',
    #     n_jobs=-1
    # ),
    
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    ),
    # "Gradient Boosting": GradientBoostingClassifier(
    #     n_estimators=500,
    #     max_depth=5,
    #     learning_rate=0.05,
    #     random_state=42
    # ),
    
    # "Neural Network": MLPClassifier(
    #     hidden_layer_sizes=(64, 32),
    #     max_iter=500,
    #     random_state=42,
    #     early_stopping=True
    # ),
}

print(f"\n{'='*60}")
print(f"Comparing {len(models)} models")
print(f"{'='*60}\n")


# ============================================
# 3. RANKING METRICS
# ============================================
def precision_at_k(labels, k=10):
    return sum(labels[:k]) / k

def recall_at_k(labels, k=10):
    total_pos = sum(labels)
    return sum(labels[:k]) / total_pos if total_pos > 0 else 0

def map_at_k(labels, k=10):
    hits = 0
    score = 0.0
    for i in range(k):
        if labels[i] == 1:
            hits += 1
            score += hits / (i + 1)
    return score / k

def ndcg_at_k(labels, k=10):
    dcg = sum([(1 / math.log2(i + 2)) if labels[i] == 1 else 0
               for i in range(k)])
    ideal = sum([(1 / math.log2(i + 2)) 
                 for i in range(min(sum(labels), k))])
    return dcg / ideal if ideal > 0 else 0

def evaluate_ranking(model, test_df, features):
    """Evaluate model on ranking metrics"""
    all_p10, all_r10, all_map10, all_ndcg10 = [], [], [], []
    
    for job_id, group in tqdm(test_df.groupby("job_id"), desc="Evaluating"):
        probs = model.predict_proba(group[features])[:, 1]
        group = group.copy()
        group["score"] = probs
        group = group.sort_values("score", ascending=False)
        
        labels = group["label"].tolist()
        
        all_p10.append(precision_at_k(labels, 10))
        all_r10.append(recall_at_k(labels, 10))
        all_map10.append(map_at_k(labels, 10))
        all_ndcg10.append(ndcg_at_k(labels, 10))
    
    return {
        "P@10": np.mean(all_p10),
        "R@10": np.mean(all_r10),
        "MAP@10": np.mean(all_map10),
        "NDCG@10": np.mean(all_ndcg10)
    }


# ============================================
# 4. TRAIN AND EVALUATE ALL MODELS
# ============================================
results = []

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    
    # Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    # Predict
    t0 = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - t0
    
    # Classification metrics
    print("\n--- Classification Metrics ---")
    print(classification_report(y_test, y_pred, digits=4))
    
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
    # Ranking metrics
    print("\n--- Ranking Metrics ---")
    ranking_metrics = evaluate_ranking(model, test_df, FEATURES)
    
    for metric, value in ranking_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Store results
    results.append({
        "Model": name,
        "Train Time (s)": train_time,
        "Inference Time (s)": inference_time,
        "ROC-AUC": auc,
        "Avg Precision": ap,
        **ranking_metrics
    })
    
    # Save model
    model_path = f"../models/model_{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, model_path)
    print(f"\nSaved: {model_path}")

print(f"\n{'='*60}")
print("All models trained!")
print(f"{'='*60}\n")


# ============================================
# 5. COMPARISON TABLE
# ============================================
results_df = pd.DataFrame(results)
results_df = results_df.round(4)

print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# Save results
results_df.to_csv("../data/model_comparison_results.csv", index=False)
print("\nSaved: ../data/model_comparison_results.csv")

# ============================================
# 6. VISUALIZATIONS
# ============================================
print("\nGenerating visualizations...")

# Plot 1: Ranking Metrics Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ["P@10", "R@10", "MAP@10", "NDCG@10"]

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    data = results_df.sort_values(metric, ascending=False)
    
    bars = ax.barh(data["Model"], data[metric])
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f"{metric} Comparison", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, (model, value) in enumerate(zip(data["Model"], data[metric])):
        ax.text(value + 0.01, i, f'{value:.3f}', 
                va='center', fontsize=10)

plt.tight_layout()
plt.savefig("../data/model_comparison_ranking.png", dpi=300, bbox_inches='tight')
print("Saved: ../data/model_comparison_ranking.png")
plt.close()


# Plot 2: Classification Metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC-AUC
ax = axes[0]
data = results_df.sort_values("ROC-AUC", ascending=False)
bars = ax.barh(data["Model"], data["ROC-AUC"])
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(data)))
for bar, color in zip(bars, colors):
    bar.set_color(color)
ax.set_xlabel("ROC-AUC", fontsize=12)
ax.set_title("ROC-AUC Comparison", fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, (model, value) in enumerate(zip(data["Model"], data["ROC-AUC"])):
    ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=10)

# Average Precision
ax = axes[1]
data = results_df.sort_values("Avg Precision", ascending=False)
bars = ax.barh(data["Model"], data["Avg Precision"])
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(data)))
for bar, color in zip(bars, colors):
    bar.set_color(color)
ax.set_xlabel("Average Precision", fontsize=12)
ax.set_title("Average Precision Comparison", fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, (model, value) in enumerate(zip(data["Model"], data["Avg Precision"])):
    ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig("../data/model_comparison_classification.png", dpi=300, bbox_inches='tight')
print("Saved: ../data/model_comparison_classification.png")
plt.close()

# Plot 3: Training Time vs Performance
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    results_df["Train Time (s)"], 
    results_df["NDCG@10"],
    s=200,
    c=range(len(results_df)),
    cmap='viridis',
    alpha=0.7,
    edgecolors='black',
    linewidth=1.5
)

for idx, row in results_df.iterrows():
    ax.annotate(
        row["Model"],
        (row["Train Time (s)"], row["NDCG@10"]),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold'
    )

ax.set_xlabel("Training Time (seconds)", fontsize=12)
ax.set_ylabel("NDCG@10", fontsize=12)
ax.set_title("Training Time vs. Ranking Quality", fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../data/model_comparison_efficiency.png", dpi=300, bbox_inches='tight')
print("Saved: ../data/model_comparison_efficiency.png")
plt.close()

print("\n✅ Model comparison complete!")
print(f"\nBest model by NDCG@10: {results_df.loc[results_df['NDCG@10'].idxmax(), 'Model']}")
print(f"Best model by ROC-AUC: {results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']}")
