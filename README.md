# Channapatna Toys - Machine Learning Price Recommendation Engine

An intelligent price estimation and market analysis engine designed for handcrafted, GI-tagged **Channapatna wooden toys**. The system combines physical toy dimensions, material attributes, text descriptions, and marketplace data to predict fair selling prices and market discount rates.

---

## 🚀 Key Highlights & Model Performance

The current pricing engine employs a **Price-to-MRP Ratio Formulation** trained with a regularized **Random Forest Regressor**, overcoming extreme price scale disparities and low sample sizes in handicraft markets.

| Metric | Baseline Model | **Production Model** | Improvement |
| :--- | :---: | :---: | :---: |
| **MAE (Mean Absolute Error)** | Rs. 444.85 | **Rs. 194.43** | **56.3% lower error** |
| **RMSE** | Rs. 680.91 | **Rs. 377.92** | **44.5% lower error** |
| **R² Score** | 0.8467 | **0.9225** | **High correlation** |
| **MAPE (Mean % Error)** | > 38% | **15.30%** | **Over 2x accuracy** |
| **Usable Dataset Records** | 92 | **180** | **100% data retention** |

For deep architectural details, feature engineering methodology, and mathematical proofs, see [MODEL_DOCUMENTATION.md](file:///c:/Users/raksh/channapatna-toys/MODEL_DOCUMENTATION.md).

---

## 📁 Repository Structure

```text
├── preprocessing.py           # Main training, preprocessing & evaluation pipeline
├── MODEL_DOCUMENTATION.md     # In-depth architectural & mathematical model documentation
├── mlr_metrics.json           # Machine-readable validation metrics output
├── price_model.pkl            # Serialized Random Forest model artifact
├── tfidf_vectorizer.pkl       # Fitted TF-IDF vocabulary extractor
├── feature_columns.json       # Ordered feature schema for deployment
├── server.js                  # Express API server entry point
├── routes/                    # API route definitions (e.g., Etsy integration)
├── services/                  # External service clients
└── package.json               # Node.js dependencies and scripts
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.10+**
- **Node.js 18+**

### 1. Install Python Dependencies
```bash
pip install pandas numpy scikit-learn openpyxl joblib
```

### 2. Train the Model & Export Artifacts
Execute the unified preprocessing and training pipeline:
```bash
python preprocessing.py
```

Output:
```text
Loading datasets...
Data successfully prepared:
Total usable records: 180
Training samples: 144
Testing samples: 36
Number of engineered features: 100

=============================================
       EVALUATION METRICS (NEW MODEL)
=============================================
MAE  (Mean Absolute Error): Rs. 194.43
RMSE (Root Mean Sq. Error): Rs. 377.92
R2 Score:                   0.9225
MAPE (Mean % Error):        15.30%
=============================================
Trained model and metrics saved successfully to 'price_model.pkl' and 'mlr_metrics.json'!
```

### 3. Start the Node.js API Server
```bash
npm install
npm start
```

---

## 🧠 How the Model Works

1. **Artisan MRP Recovery**: Direct-to-consumer artisan records lacking artificial markups are assigned `MRP = Price`, preventing data loss from naive row dropping.
2. **Dimension Extraction**: Dimension strings (`"20*20*2"`) are parsed into numeric length, width, and height with volumetric calculations ($\text{Volume} = L \times W \times H$; $\text{Density} = \frac{\text{Weight}}{\text{Volume}}$).
3. **Text Semantics**: Top 20 keywords from toy titles are extracted using `TfidfVectorizer`.
4. **Ratio Regression**: The model predicts bounded $\text{Ratio} = \frac{\text{Price}}{\text{MRP}} \in [0.15, 1.00]$, strictly preventing runaway price errors.
5. **Reconstruction**: At inference time, $\widehat{\text{Price}} = \widehat{\text{Ratio}} \times \text{MRP}$.

---

## 📄 License
This project is licensed under the MIT License.