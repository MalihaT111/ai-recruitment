# Model Comparison Notebook

This notebook compares multiple ML models for resume-job matching.

## Quick Start

```bash
# Run the comparison script
cd notebooks
python model_comparison.py
```

## Models Compared

1. **Logistic Regression** - Fast baseline, linear decision boundary
2. **Random Forest** - Ensemble of decision trees, handles non-linearity
3. **Gradient Boosting** - Sequential ensemble, often best performance
4. **SVM (RBF)** - Non-linear kernel, good for complex patterns
5. **Neural Network** - Deep learning, can learn complex interactions

## Evaluation Metrics

### Classification Metrics
- **ROC-AUC**: Overall discrimination ability
- **Average Precision**: Precision-recall trade-off

### Ranking Metrics (More Important!)
- **Precision@10**: % of top-10 that are relevant
- **Recall@10**: % of relevant items in top-10
- **MAP@10**: Mean Average Precision at 10
- **NDCG@10**: Normalized Discounted Cumulative Gain (best metric for ranking)

## Expected Results

Based on similar tasks, you should see:
- **Gradient Boosting**: Highest NDCG@10 (~0.88-0.90)
- **Random Forest**: Close second (~0.86-0.88)
- **Logistic Regression**: Fast but lower performance (~0.82-0.85)
- **Neural Network**: Good but may overfit (~0.85-0.87)
- **SVM**: Slowest, moderate performance (~0.83-0.86)

## Outputs

After running, you'll get:
- `../data/model_comparison_results.csv` - Full results table
- `../models/model_*.pkl` - Saved models
- `../data/model_comparison_ranking.png` - Ranking metrics visualization
- `../data/model_comparison_classification.png` - Classification metrics
- `../data/model_comparison_efficiency.png` - Time vs. performance

## Next Steps

1. **Hyperparameter Tuning**: Use GridSearchCV on best model
2. **Feature Engineering**: Add more features
3. **Ensemble**: Combine top 2-3 models
4. **Learning-to-Rank**: Try LightGBM Ranker or XGBoost Ranker
