# 🧠 Autism Detection System (Machine Learning + Flask)

A production-quality, beginner-friendly Machine Learning web application that
predicts the likelihood of Autism Spectrum Disorder (ASD) from behavioural and
demographic inputs. Built with **Python, Flask, Scikit-learn, HTML, CSS, JS**.

Perfect for **internships, final-year projects, and ML portfolios**.

---

## ✨ Features

- 🎯 Three ML models compared: **Logistic Regression, Random Forest, Decision Tree**
- 📊 Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- 📈 Auto-generated model comparison chart
- 🧾 PDF prediction report download (ReportLab)
- 🌙 Dark-mode toggle, loading animation, input validation
- 📱 Fully responsive modern healthcare-style UI
- 📊 Dashboard with live statistics
- 🧪 Realistic synthetic ASD-screening dataset (1,000 samples)

---

## 📁 Project Structure

```
autism_app/
├── app.py                  # Flask backend
├── model_training.py       # Train & evaluate ML models
├── dataset.csv             # Training dataset
├── autism_model.pkl        # Trained model (best of 3)
├── requirements.txt
├── README.md
├── templates/
│   ├── index.html          # Home + prediction form
│   ├── result.html         # Prediction result + recommendations
│   └── about.html          # About / dashboard
└── static/
    ├── style.css
    ├── script.js
    └── model_comparison.png
```

---

## 🚀 How to Run (PyCharm / VS Code)

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the ML model (creates autism_model.pkl + chart)
python model_training.py

# 4. Start the Flask server
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 🧪 Input Fields

| Field | Type |
|-------|------|
| Age | number (1–100) |
| Gender | male / female |
| Family history of autism | yes / no |
| Speech delay | yes / no |
| Social interaction difficulty | yes / no |
| Repetitive behavior | yes / no |
| Eye contact | normal / poor |
| Learning difficulty | yes / no |
| Anxiety level | 0–10 |
| Communication score | 0–10 |

---

## 📊 Output

- **Prediction:** Autism / Non-Autism
- **Confidence percentage**
- **Personalised recommendations**
- **Downloadable PDF report**

---

## ⚠️ Disclaimer

This tool is for **educational and screening-awareness purposes only**.
It is **NOT a medical diagnosis**. Please consult a licensed clinician.
