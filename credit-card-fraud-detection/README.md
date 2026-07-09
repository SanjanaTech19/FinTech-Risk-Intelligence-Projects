Fintech Risk Intelligence: Fraud Detection API

A scalable, containerized FastAPI application designed for real-time credit card fraud detection. This project demonstrates end-to-end MLOps capability, integrating robust data validation, model inference, and modular architecture.

🚀 Project Overview

This service provides low-latency fraud probability scores for financial transactions. By utilizing FastAPI for high-performance serving and Pydantic for rigorous data validation, the system ensures that only correctly formatted transaction data is processed by the underlying machine learning model.

🛠 Tech Stack

API Framework: FastAPI

Model: LightGBM (Gradient Boosting)

Validation: Pydantic (Modular schemas)

Containerization: Docker

Version Control: Git

📂 Project Structure
Plaintext
├── app/
│   ├── __init__.py          # Initializes the app package
│   ├── main.py              # API routes and inference logic
│   └── schemas.py           # Data blueprints (TransactionPayload)
├── data/                    # Placeholder for datasets (.gitkeep)
├── models/                  # ML artifacts (ignored by git)
├── .gitignore               # Excludes secrets, caches, and large data
├── Dockerfile               # Production-ready environment build
├── README.md                # Project documentation
└── requirements.txt         # Project dependencies


⚙️ Installation & Usage

1. Clone the Repository
Bash
git clone https://github.com/SanjanaTech19/FinTech-Risk-Intelligence-Projects.git
cd FinTech-Risk-Intelligence-Projects

2. Build the Docker Container
Bash
docker build -t fraud-detection-api .

3. Run the API
Bash
docker run -p 8000:8000 fraud-detection-api

4. API Documentation
Once running, navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to access the interactive Swagger UI for testing your endpoints.

📡 API Endpoint Example

To test the prediction endpoint, send a POST request to /predict:

JSON
{
  "Time": 100.0,
  "Amount": 50.0,
  "V_features": [0.0, 0.0, ..., 0.0] 
}

Note: V_features must contain exactly 28 numeric items.