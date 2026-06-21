# TruthLens - AI Powered News Credibility Analyzer

TruthLens is an AI-powered web application that evaluates the credibility of news articles using a hybrid Machine Learning and Deep Learning pipeline. The system combines a fine-tuned **DistilBERT** model with a **Logistic Regression + TF-IDF** classifier to generate a credibility score (0–100). It also integrates the **Google Fact Check API** to provide additional verification, helping users assess the reliability of online news content.

---

## Features

* 📰 AI-powered news credibility analysis
* 🤖 Fine-tuned DistilBERT for contextual understanding
* 📊 Logistic Regression + TF-IDF based classification
* 🌐 Google Fact Check API integration
* 📈 Credibility score ranging from **0–100**
* 👤 User authentication system
* 🔐 Secure OTP-based password reset
* 📩 Contact Us form with database storage
* 📱 Responsive and user-friendly interface

---

## Tech Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Flask
* SQLAlchemy
* Flask-Login
* Flask-Mail

### Machine Learning & Deep Learning

* Fine-tuned DistilBERT
* Logistic Regression
* TF-IDF Vectorizer
* spaCy
* NumPy
* Pandas
* Scikit-learn
* Transformers
* PyTorch

### Database

* SQLite

---

## Fine-tuned DistilBERT Model

Download the fine-tuned DistilBERT model from the link below:

**Model:**
https://drive.google.com/file/d/1-NgDal2jM3q-9vJWl86AZwZnmNcrdY-H/view?usp=sharing


---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd TruthLens
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add the required environment variables:

```text
SECRET_KEY=
DATABASE_URL=
EMAIL=
APP_PASSWORD=
FACT_CHECK_API_KEY=
ADMIN_EMAIL=
```

### 5. Run the application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

⭐ If you found this project useful, consider giving this repository a **Star**. Your support is greatly appreciated!
