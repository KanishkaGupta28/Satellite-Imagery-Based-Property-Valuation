# Satellite Imagery-Based Property Valuation 

This project predicts residential property prices by combining **structured housing attributes** with **satellite imagery** using a **multimodal machine learning pipeline**.  
By integrating visual context from satellite images, the model captures greenery, water bodies, and urban density — improving prediction accuracy beyond traditional tabular models.

---

## Project Structure

```
Satellite-Imagery-Based-Property-Valuation/
│
├── scripts/
│   └── data_fetcher.py              # Script to download satellite images
│
├── notebooks/
│   ├── preprocessing.ipynb          # Data cleaning & feature engineering
│   └── model_training.ipynb         # Model training & evaluation
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── prediction.csv                   # Final price predictions
├── requirements.txt                 # Python dependencies
└── README.md
```

**Note:** Folders like `images/`, `venv/`, and `artifacts/` are ignored and not included in the repository.  
Images are generated programmatically via `data_fetcher.py`.

---

## Dataset Description

### Tabular Features
- Bedrooms, Bathrooms, Living area (`sqft_living`), Lot size (`sqft_lot`)  
- Floors, Condition & Grade  
- Waterfront and View indicators  
- Nearby living and lot area statistics (`sqft_living15`, `sqft_lot15`)  

**Target variable:** `price` (transformed with `log1p(price)` for stability)

### Image Data
- Satellite images corresponding to each property  
- Capture environmental context: greenery, water bodies, and urban/road density  

---

## Deliverables

1. **Prediction File (CSV)**  
   - `prediction.csv` with columns: `id, predicted_price`

2. **Code Repository**  
   - `scripts/data_fetcher.py`  
   - `notebooks/preprocessing.ipynb`  
   - `notebooks/model_training.ipynb`  
   - `README.md`

3. **Project Report (PDF)**  
   - Overview, EDA, visual insights, architecture diagram, and results  

---

## Modeling Approach

### 1. Tabular Data Model
- Missing value imputation, feature scaling  
- Trained **XGBoost Regressor** using only structured data  

### 2. Image Feature Extraction
- CNN Backbone: **ResNet-18 (pretrained on ImageNet)**  
- Extracted **512-dimensional embeddings** from satellite images using PyTorch  

### 3. Multimodal Fusion
- Concatenated tabular features with image embeddings  
- Trained final **XGBoost multimodal model**  
- Performance compared with tabular-only baseline  

---

## Model Explainability

- **Grad-CAM** applied to ResNet-18 to visualize image regions influencing predictions  
- Typical high-impact regions include:  
  - Vegetation & greenery  
  - Water bodies  
  - Urban density and road networks  

This improves the transparency and interpretability of the visual model.

---

## How to Run

### 1. Clone the Repository
```
git clone https://github.com/KanishkaGupta28/Satellite-Imagery-Based-Property-Valuation.git
cd Satellite-Imagery-Based-Property-Valuation
```

### 2. Create Virtual Environment
```
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Download Satellite Images
```
python scripts/data_fetcher.py
```

### 5. Run Jupyter Notebooks (in order)
1. `notebooks/preprocessing.ipynb`  
2. `notebooks/model_training.ipynb`

---

## Results

| Model                       | RMSE     | R² Score   |
|------------------------------|----------|-------------|
| Tabular Only                 | Baseline | Moderate    |
| Tabular + Satellite Imagery  | Improved | Higher      |

The multimodal model outperformed the baseline, showing that **visual contextual cues significantly enhance** property price predictions.

---

## Tech Stack
- **Data Handling:** Pandas, NumPy, GeoPandas  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Deep Learning:** PyTorch, torchvision  
- **Image Processing:** PIL, OpenCV  
- **Visualization:** Matplotlib, Seaborn  
- **Explainability:** Grad-CAM  

---

## Future Improvements
- Fine-tune CNN on domain-specific satellite image datasets  
- Use higher-resolution image tiles  
- Explore attention-based multimodal fusion techniques  
- Integrate geospatial coordinates for better environmental context  

---

## Author
**Kanishka Gupta**  
This project was developed as part of a multimodal machine learning evaluation combining **computer vision** and **structured data** for real-estate analytics.
```

