# 💧 Water Quality Prediction using Machine Learning

This project predicts whether water is **potable (safe to drink)** using various physicochemical parameters. A machine learning model is trained on real-world data to assist in identifying drinkable water sources efficiently.

---

## 📂 Project Contents

- `WATER_QUALITY_PREDECTION.ipynb` — Full notebook with data preprocessing, EDA, training, and evaluation.
- `water_potability.csv` — Dataset with water quality features and potability labels.
- `Water_Quality_ML_Trained_Model.sav` — Saved ML model for deployment/inference.

---

## 📊 Dataset Overview

The dataset (`water_potability.csv`) contains the following features:

| Feature        | Description                               |
|----------------|-------------------------------------------|
| pH             | pH value of water                         |
| Hardness       | Hardness of water                         |
| Solids         | Total dissolved solids                    |
| Chloramines    | Amount of chloramines                     |
| Sulfate        | Sulfate concentration                     |
| Conductivity   | Conductivity of water                     |
| Organic_carbon | Organic carbon content                    |
| Trihalomethanes| Trihalomethanes level                     |
| Turbidity      | Water turbidity                           |
| Potability     | Target variable (0 = Not drinkable, 1 = Drinkable) |

---

## 🧠 Machine Learning Models Used

The notebook explores multiple models. The final model was chosen based on accuracy and F1-score:

- Logistic Regression
- Random Forest ✅ (Best Performance)
- Decision Tree
- Support Vector Machine
- K-Nearest Neighbors

---

## 🚀 How to Run

1. Clone this repository:
```bash
git clone https://github.com/your-username/water-quality-prediction.git
cd water-quality-prediction
```

2. Install the required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

3. Run the Jupyter Notebook:
```bash
jupyter notebook WATER_QUALITY_PREDECTION.ipynb
```

4. (Optional) Load the trained model in Python:
```python
import joblib
model = joblib.load("Water_Quality_ML_Trained_Model.sav")
```

---

## 📈 Model Accuracy

- ✅ Random Forest Classifier:
  - **Accuracy:** ~79%
  - **F1-Score:** ~0.76
  - Robust to missing data and feature scaling
- Models were evaluated using a confusion matrix, accuracy, precision, and recall.

---

## 🔬 Exploratory Data Analysis (EDA)

- Checked for missing values and handled them
- Visualized distributions and correlations
- Observed important relationships (e.g., pH vs potability, organic carbon)

---

## 📦 Future Enhancements

- Add a web interface using Flask or Streamlit
- Integrate real-time water quality sensors
- Deploy as an API or Android app
- Perform feature engineering or use advanced ML models (e.g., XGBoost)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Dataset from Kaggle: [Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- Built with ❤️ using Python and Machine Learning
