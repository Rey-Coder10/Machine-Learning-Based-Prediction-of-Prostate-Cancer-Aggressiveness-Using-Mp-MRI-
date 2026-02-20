═══════════════════════════════════════════════════════════════════
  PROSTATE CANCER ML PIPELINE — VISUAL DEMONSTRATION SNAPSHOTS
═══════════════════════════════════════════════════════════════════

📌 PURPOSE:
These screenshots demonstrate the complete machine learning pipeline
for viva/presentation. Each image captures a key step with clear 
labels, metrics, and professional formatting.

═══════════════════════════════════════════════════════════════════
📸 SNAPSHOT GUIDE
═══════════════════════════════════════════════════════════════════

01_Dataset_Overview.png
├─ Dataset specifications table (source, size, features)
├─ Class distribution pie chart (42% low-grade, 58% high-grade)
└─ CSV data preview (first 5 patients)
   ▸ Shows: 300 patients, 1200 features, PROSTATEx dataset

02_Train_Val_Test_Split.png
├─ Split visualization (70:15:15 ratio)
├─ Patient count breakdown per set
└─ Stratification verification (class distribution maintained)
   ▸ Shows: Train=210, Val=45, Test=45

03_Feature_Selection_Pipeline.png
├─ Complete flow diagram (Original → Standardized → LASSO → PCA)
├─ Dimensionality reduction bar chart
└─ PCA explained variance curve (95% threshold)
   ▸ Shows: 1200 → 37 → 33 features (97% reduction)

04_Model_Training_Hyperparameters.png
├─ 4 model cards with specifications
├─ Hyperparameter details for each model
└─ Training status confirmation
   ▸ Shows: SVM, RF, LR, XGBoost configurations

05_Results_Summary_Table.png
├─ Complete metrics table (all 4 models)
├─ Gold highlighting for best performance
└─ Best model annotation (SVM, AUC 0.93)
   ▸ Shows: Accuracy, Precision, Recall, F1, AUC

06_Confusion_Matrices.png
├─ All 4 confusion matrices (2×2 grid)
├─ TP/TN/FP/FN labels clearly marked
└─ Color-coded by model performance
   ▸ Shows: Actual vs Predicted classification

07_ROC_Curves.png
├─ All 4 ROC curves on single plot
├─ AUC values for each model
└─ Best model highlighted (SVM: 0.93)
   ▸ Shows: TPR vs FPR comparison

08_Key_Code_Snippets.png
├─ 6 essential code blocks with syntax highlighting
├─ Data loading, splitting, LASSO, PCA, training, evaluation
└─ Clean, readable format
   ▸ Shows: Actual Python implementation

═══════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════
✅ QUICK REFERENCE
═══════════════════════════════════════════════════════════════════

Dataset Size:        300 patients
Features (Original): 1200
Features (Final):    33 (after LASSO + PCA)
Train:Val:Test:      210:45:45 (70:15:15)
Best Model:          SVM (RBF kernel)
Best AUC:            0.93
Best Accuracy:       91%

Models Compared:
  ✓ SVM (Support Vector Machine)
  ✓ Random Forest
  ✓ Logistic Regression  
  ✓ XGBoost

Key Techniques:
  ✓ LASSO (L1 regularization) → 1200 to 37 features
  ✓ PCA (95% variance) → 37 to 33 components
  ✓ Stratified sampling → Balanced splits
  ✓ ROC-AUC evaluation → Threshold-independent

═══════════════════════════════════════════════════════════════════
📧 NOTES
═══════════════════════════════════════════════════════════════════

• All snapshots are HIGH RESOLUTION (150 DPI) — print-ready
• Professional color scheme — medical/research theme
• Clear labels and annotations — easy to explain
• Includes both visual and code representations
• Covers entire pipeline from data to deployment-ready model

Generated on: 2026-02-02
Project: Machine Learning Based Prediction of Prostate Cancer Aggressiveness Prediction using mp-MRI
Authors: Divyendu Kumar Mishra, Yadav Rahul Suresh Chandra(23001309007),M.Tech(CSE)
University: Veer Bahadur Singh Purvanchal University, Jaunpur

═══════════════════════════════════════════════════════════════════
