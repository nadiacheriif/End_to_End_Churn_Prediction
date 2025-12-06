import gradio as gr
from gradio.blocks import Blocks
import requests

API_URL_SINGLE = "http://localhost:8000/predict/single"

def predict_single_ui(customerID, gender, senior_citizen, tenure, monthly_charges, total_charges):
    payload = {
        "data": {
            "customerID": customerID,
            "gender": gender,
            "SeniorCitizen": int(senior_citizen),
            "tenure": int(tenure),
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }
    }

    try:
        response = requests.post(API_URL_SINGLE, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------------------
# Gradio 5.x UI using Blocks
# --------------------------------------------------------------
with Blocks(title="📡 Telco Customer Churn Prediction") as demo:
    
    gr.Markdown("# 📡 Telco Customer Churn Prediction")
    gr.Markdown("Enter customer attributes to predict the probability of churn.")

    with gr.Row():
        customerID = gr.Textbox(label="Customer ID", placeholder="1234-ABCD")
        gender = gr.Dropdown(["Male", "Female"], label="Gender")

    with gr.Row():
        senior_citizen = gr.Checkbox(label="Senior Citizen (1 = Yes, 0 = No)")
        tenure = gr.Number(label="Tenure (months)")

    with gr.Row():
        monthly_charges = gr.Number(label="Monthly Charges")
        total_charges = gr.Number(label="Total Charges")

    output = gr.JSON(label="Prediction Result")
    
    submit_btn = gr.Button("Predict")

    submit_btn.click(
        fn=predict_single_ui,
        inputs=[customerID, gender, senior_citizen, tenure, monthly_charges, total_charges],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
