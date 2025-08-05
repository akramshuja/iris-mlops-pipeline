
# 🌸 End-to-End MLOps Pipeline for Iris Classification

This repository provides a complete, production-grade MLOps pipeline for the classic Iris dataset. It showcases best practices in model development, deployment, and monitoring using modern tools and frameworks.

---

## ✨ Features

- **📊 Experiment Tracking**: MLflow logs and compares all training runs.
- **📦 Model Registry**: Centralized versioning and model lifecycle management via MLflow Model Registry.
- **📁 Data Versioning**: Uses DVC to version datasets, ensuring reproducibility.
- **🚀 REST API**: Serves predictions using FastAPI for high performance and scalability.
- **🐳 Containerization**: Dockerized services for consistent deployment.
- **🔁 CI/CD Automation**: GitHub Actions pipeline tests, builds, and deploys the model and API.
- **📈 Monitoring**: Prometheus tracks key metrics and logs all prediction requests.

---

## 🛠️ Tech Stack

| Purpose              | Tool(s)               |
|----------------------|------------------------|
| Version Control       | Git, GitHub             |
| Data Versioning       | DVC                    |
| Experiment Tracking   | MLflow                 |
| Model Serving         | FastAPI                |
| Containerization      | Docker, Docker Compose |
| CI/CD                 | GitHub Actions         |
| Monitoring            | Prometheus             |

---

## 🚀 Getting Started

### ✅ Prerequisites

Ensure you have the following installed:

- Git
- Python 3.9+
- Docker & Docker Compose

---

### 🔧 Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/akramshuja/iris-mlops-pipeline.git
   cd iris-mlops-pipeline
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull the Data using DVC**
   ```bash
   dvc pull
   ```

---

## 🏃 Running the Application (Local Hybrid Mode)

This project uses a hybrid setup: MLflow runs on the host machine, while the API and monitoring tools run in Docker containers.

---

### ▶️ Step 1: Start the MLflow Server

In one terminal, run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0
```

Access the MLflow UI at: [http://localhost:5000](http://localhost:5000)

---

### 🧠 Step 2: Train and Register the Model

In a new terminal:

```bash
python scripts/train.py
python scripts/register_model.py
```

These scripts log the trained model and register it in MLflow’s Model Registry.

---

### 🌐 Step 3: Start the API and Monitoring Services

Start FastAPI and Prometheus using Docker Compose:

```bash
docker-compose up --build
```

The FastAPI app will be available at: [http://localhost:8000](http://localhost:8000)  
Prometheus dashboard: [http://localhost:9090](http://localhost:9090)

---

## 📦 CI/CD with GitHub Actions

This project includes a GitHub Actions workflow that:

- Runs automated tests
- Builds the Docker image
- Pushes it to Docker Hub (once configured)

---

## 📈 Monitoring & Logging

- **Prediction Logs**: All API calls are logged for traceability.
- **Prometheus**: Collects metrics like request counts, latency, etc.

---

## 📂 Project Structure

```
iris-mlops-pipeline/
│
├── .github/workflows/        # CI/CD workflows
├── api/                      # FastAPI app
├── data/                     # Raw and processed datasets
├── dvc.yaml                  # DVC pipeline config
├── docker-compose.yml        # Compose file to start services
├── mlruns/                   # MLflow experiment data
├── models/                   # Saved models
├── scripts/                  # Training & registration scripts
├── requirements.txt
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🙌 Acknowledgements

- [scikit-learn](https://scikit-learn.org/)
- [MLflow](https://mlflow.org/)
- [DVC](https://dvc.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Prometheus](https://prometheus.io/)

---

Feel free to fork, star ⭐, or contribute to improve this pipeline!
