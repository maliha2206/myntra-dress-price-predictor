# 🌸 Floral Dress Price Predictor

A Machine Learning web application that predicts the price of floral dresses based on product attributes such as length, neck type, sleeve style, material, and category.

Built using **Python, Scikit-learn, XGBoost, and Streamlit**, this project demonstrates end-to-end ML workflow including data scraping, preprocessing, feature engineering, model training, and deployment.

---

## 🚀 Features

* Predicts dress price based on input attributes
* Uses **XGBoost model** for accurate predictions
* Interactive UI built with Streamlit
* Cleaned and enhanced dataset for better performance
* Real-world dataset scraped from Myntra

---

## 📂 Project Structure

```
├── app.py                              # Streamlit application
├── xgboost.pkl                         # Trained XGBoost model
├── ui_preprocessor.pkl                 # Preprocessing pipeline
├── cleaned_ethnic_dress_dataset_for_deployment.csv
├── final_enhanced_dataset.csv
├── myntra_floral_195_FINAL.csv
├── model_metadata.json
├── requirements.txt
├── README.md
```

---

## ⚙️ Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Selenium (for data scraping)

---

## 📊 Dataset

* Data scraped from Myntra floral dress listings
* Includes features like:

  * Price
  * Rating
  * Reviews
  * Length
  * Neck type
  * Sleeve style
  * Material
  * Category (Casual, Formal, Party Wear, etc.)

---

## 🧠 Model

* Model Used: **XGBoost Regressor**
* Preprocessing:

  * Label encoding
  * Feature transformation using pipeline
* Target:

  * Dress price (in INR ₹)

---

## ▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/maliha-22/myntra-dress-price-predictor.git
cd floral-dress-price-predictor
2. Create virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run the app
streamlit run app.py

--- 


## 💡 Usage

1. Select dress attributes (length, neck, sleeve, etc.)
2. Choose category (casual, formal, etc.)
3. Click **Predict Price**
4. Get estimated price instantly

---

## 📈 Future Improvements

* Add image-based classification (using CNN/LLM)
* Improve dataset size for better accuracy
* Deploy on cloud (Streamlit Cloud / AWS / Render)
* Add recommendation system

---

## 👩‍💻 Author

**Maliha Mubeen**
**Mariam Firdous**
**Meharunnisa Begum**
**Afia Refal**

--- 


