# 🏥 Medical Insurance Cost Prediction

This project predicts medical insurance costs based on factors such as age, BMI, smoking habits, and other relevant features. It utilizes Machine Learning techniques to estimate insurance charges accurately.

---

## 🚀 Features

- Predicts insurance costs based on user input.
- Utilizes **Random Forest** and **Linear Regression** models.
- Interactive **Streamlit** web app for user-friendly interaction.
- Trained on a dataset containing health-related attributes.
- Supports model performance comparison.
- Visualizes data distributions and feature correlations.
- Provides insights into how different factors affect insurance costs.

---

## 📊 Tech Stack

- **Python**: `pandas`, `NumPy`, `scikit-learn`
- **Machine Learning**: `Random Forest`, `Linear Regression`
- **Streamlit**: (for interactive UI)
- **Matplotlib & Seaborn**: (for data visualization)
- **Joblib**: (for model saving and loading)

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/medical-insurance-cost-prediction.git
   cd medical-insurance-cost-prediction
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎯 Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Input required features** such as age, BMI, smoking status, and region.
3. **Get an estimated insurance cost** based on the trained machine learning model.

---

## 🏋️‍♂️ Model Training

To retrain the model with updated data:
```bash
python train.py
```
This will generate a new model file (`MIPML.pkl`).

---

## 📈 Model Performance

- **Random Forest**: Provides higher accuracy but is computationally expensive.
- **Linear Regression**: Simpler and faster but may not capture complex relationships.
- Performance evaluation metrics: **R² Score**, **Mean Squared Error (MSE)**.

---

## 📝 Dataset

The dataset includes:
- `age`: Age of the person.
- `sex`: Gender of the insured individual.
- `bmi`: Body Mass Index (BMI).
- `children`: Number of dependent children.
- `smoker`: Smoking status (yes/no).
- `region`: Residential region.
- `charges`: Actual insurance cost.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. **Fork the repository**
2. **Create a new branch** (`git checkout -b feature-branch`)
3. **Commit changes** (`git commit -m 'Add new feature'`)
4. **Push to GitHub** (`git push origin feature-branch`)
5. **Open a Pull Request**

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For any inquiries, contact:
📧 Email: manishprajapati.cs1@gmail.com  
🔗 GitHub: Manish Kumar(https://github.com/ManishKumarCs)  
💼 LinkedIn: Manish Kumar(https://www.linkedin.com/in/manishkumarcs1)

