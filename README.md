# End-to-End Churn Classification App

## Overview

This project delivers an end-to-end machine learning solution for **customer churn prediction**, designed to help businesses **retain customers, protect recurring revenue, and improve long-term loyalty**.

Customer churn represents a major financial risk for subscription-based and service-driven companies. This system identifies customers who are **at risk of leaving** early enough to allow businesses to take **proactive retention actions** (discounts, personalized offers, customer support outreach, or service improvements).

From both a **technical and business standpoint**, the project is intentionally designed to **prioritize customer retention over pure accuracy**. In real-world scenarios, it is **less costly to mistakenly flag a loyal customer as at-risk** than to fail to detect a customer who is actually going to churn.

---

## 🌟 Key Features

- 📊 Exploratory Data Analysis (EDA) focused on churn behavior
- 🧠 Supervised machine learning models for churn classification
- ⚖️ Business-oriented evaluation strategy (recall-focused)
- 🔄 End-to-end ML pipeline (data → model → inference)
- 🗃️ Modular and production-ready project structure
- 🌐 Optional API / deployment-ready design (Flask & Docker)

---

## 📈 Business Impact & Value

This project enables organizations to:

- **Reduce customer churn** by detecting early warning signals
- **Increase customer lifetime value (CLV)**
- **Lower acquisition costs** by prioritizing retention over replacement
- **Empower marketing & customer success teams** with data-driven insights
- **Align machine learning outputs with financial risk**

### Strategic Design Choice
> It is better to detect that a client *might leave* when they actually won’t,  
> than to miss detecting a client who *will* leave.

This aligns model behavior with **real financial consequences**, where missed churn leads to permanent revenue loss.

---

## Project Structure

```
End to end Churn Classification App/
├── notebooks/                # EDA and modeling notebooks
├── src/                      # Core application code
├── Scripts/                  # Utility scripts
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container configuration
├── docker-compose.yml        # Multi-service setup
└── README.md                 # Project documentation
```

## How It Works

1. **Data Exploration**: Analyze customer data to identify churn drivers.
2. **Preprocessing**: Clean and prepare data (handling missing values, encoding, scaling).
3. **Feature Engineering**: Create predictive features.
4. **Model Training**: Train and compare classification models.
5. **Evaluation**: Use recall-focused metrics to prioritize churn detection.
6. **Deployment**: Serialize models for production use.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Steps
1. Clone the repository:
   ```bash
   git clone <https://github.com/nadiacheriif/End_to_End_Churn_Prediction.git>
   cd "End to end Churn Classification App"
   ```

2. Create a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python app.py
```

This starts the web app for churn prediction. Follow the on-screen instructions to input customer data and get predictions.

## Model Evaluation Philosophy

Traditional accuracy isn't ideal for churn prediction. This project focuses on minimizing false negatives (missed churn) over false positives (unnecessary outreach), as undetected churn leads to permanent revenue loss.

## Business Use Cases

- **Telecommunications**: Prevent competitor switches.
- **SaaS Platforms**: Retain subscribers proactively.
- **Banking/Insurance**: Identify closure risks.
- **E-commerce**: Stabilize subscription revenue.

Integrate predictions into CRM, marketing tools, or dashboards.

## Key Takeaways

- Churn prediction is critical for business sustainability.
- ML models should reflect financial realities.
- Retention strategies drive long-term value.