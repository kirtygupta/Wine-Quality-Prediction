# 🍷 Wine Quality Prediction

Predicting the quality of red wine using machine learning techniques based on physicochemical tests.

---

## 📌 Objective

The goal of this project is to build a machine learning model that can predict whether a red wine is of good or bad quality, based on its chemical attributes such as acidity, sugar, pH, alcohol content, and more.

---

## 📁 Dataset

* **Source**: [Kaggle - Red Wine Quality Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download)
* **Size**: 1599 samples, 12 columns
* **Target Variable**: `quality` (integer score from 0 to 10)

---

## 📚 Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

---

## 🔍 Data Overview

* 12 columns including features like:

  * `fixed acidity`, `volatile acidity`, `citric acid`
  * `residual sugar`, `chlorides`, `pH`, `alcohol`
* No missing values.
* Target is binarized: wines with quality >= 7 are labeled as **good** (1), otherwise **bad** (0).

---

## 📊 Exploratory Data Analysis

* Countplot to show wine quality distribution
  <p align="center">
  <img width="300" height="300" alt="output_16_1" src="https://github.com/user-attachments/assets/5d933988-5fb7-4f17-9a74-12074b2d731b" /></p>
  
* Barplots for relationships:

  * `volatile acidity` vs `quality`
    <p align="center">
    <img width="300" height="300" alt="output_17_1" src="https://github.com/user-attachments/assets/451dc4df-2369-4681-8846-afe1036bb639" /></p>
    
  * `citric acid` vs `quality`
    <p align="center">
    <img width="300" height="300" alt="output_18_1" src="https://github.com/user-attachments/assets/04f8571f-8ce1-4e30-ba08-9a89bd17bc07" /></p>
    
* Heatmap of feature correlations
<p align="center">
  <img width="500" height="500" alt="output_21_1" src="https://github.com/user-attachments/assets/ff7f29b9-8822-47d9-a7d6-f62b333eb66f" />
</p>

---

## ⚙️ Data Preprocessing

* Features (`X`) and labels (`Y`) are separated.
* Target is binarized:

  ```python
  Y = wine_dataset['quality'].apply(lambda y: 1 if y >= 7 else 0)
  ```

---

## 🧠 Model Building

* **Algorithm Used**: `RandomForestClassifier`
* Train-test split: 80% training, 20% testing
* Model training:

  ```python
  model = RandomForestClassifier()
  model.fit(X_train, Y_train)
  ```

---

## ✅ Evaluation

* **Accuracy on test data**: `~93.1%`

  ```python
  accuracy_score(Y_test, model.predict(X_test))
  ```

---

## 🔮 Prediction Example

Predicting the quality of a sample wine:

```python
input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)
input_data_reshaped = np.asarray(input_data).reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 1:
    print("Good Quality Wine")
else:
    print("Bad Quality Wine")
```

---

## 📌 Conclusion

This notebook demonstrates the end-to-end process of wine quality prediction:

* Loading and exploring the dataset
* Data visualization and feature analysis
* Preprocessing and target engineering
* Training a classification model
* Making predictions with high accuracy

---

## 📎 Author

**Kirty Gupta**
*Machine Learning Enthusiast*
🔗 [LinkedIn](https://www.linkedin.com/in/kirtygupta111/) | [GitHub](https://github.com/kirtygupta)

---

> ⭐ *Feel free to fork, star, or contribute to improve this project!*
