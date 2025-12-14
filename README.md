# 🏨 Hotel Reservation Cancellation Prediction – End-to-End MLOps Project

## 📌 Project Overview

This project is a **production-ready end-to-end MLOps system** designed to predict whether a hotel reservation will be **Canceled** or **Not Canceled** based on historical booking data.

The project goes beyond traditional machine learning by implementing the **complete ML lifecycle**, including data ingestion from cloud storage, preprocessing, model training with experiment tracking, automated pipelines, CI/CD, containerization, and cloud deployment.

The final solution is deployed as a **Flask web application on Google Cloud Run**, making it accessible to real users.

---

## 🎯 Business Problem

Hotel reservation cancellations lead to:
- Revenue loss  
- Poor occupancy planning  
- Inefficient resource allocation  

This system helps hotels **predict cancellation risk in advance**, enabling better decision-making such as overbooking strategies, pricing optimization, and customer engagement.

---

## 🧠 Solution Approach

1. Raw booking data is stored in **Google Cloud Storage**
2. Automated **data ingestion** downloads and splits the data
3. **Data preprocessing & feature engineering** are applied
4. **LightGBM model** is trained with hyperparameter tuning
5. **MLflow** tracks experiments, metrics, and artifacts
6. A **training pipeline** orchestrates the ML workflow
7. The model is exposed via a **Flask web application**
8. Application is **Dockerized**
9. **Jenkins CI/CD** builds and pushes the image to GCR
10. Application is deployed on **Google Cloud Run**

---

## 🏗️ System Architecture


---

## 🛠️ Tech Stack

### Programming & Machine Learning
- Python
- Pandas, NumPy
- Scikit-learn
- LightGBM

### MLOps & Experiment Tracking
- MLflow

### Backend & Web
- Flask
- HTML, CSS

### Cloud & DevOps
- Google Cloud Platform (GCP)
  - Cloud Storage (GCS)
  - Container Registry (GCR)
  - Cloud Run
- Docker
- Jenkins

### Version Control
- Git
- GitHub

---

## 📂 Project Structure


---

## 🔍 Key Modules Explanation

### 1️⃣ Data Ingestion
- Downloads raw CSV data from GCP Bucket
- Splits data into train and test sets
- Uses service account authentication
- Fully configurable and logged

### 2️⃣ Data Preprocessing
- Drops unnecessary columns
- Handles categorical and numerical features
- Label encoding
- Skewness handling
- Feature selection using importance
- Saves processed datasets

### 3️⃣ Model Training
- Uses **LightGBM (LGBMClassifier)**
- Hyperparameter tuning with `RandomizedSearchCV`
- Model evaluation and persistence
- MLflow experiment logging

### 4️⃣ Experiment Tracking (MLflow)
- Tracks parameters and metrics
- Stores trained models
- Enables experiment comparison and reproducibility

### 5️⃣ Training Pipeline
- Automates ingestion → preprocessing → training
- Enables one-command retraining
- Ensures reproducibility

### 6️⃣ Flask Web Application
- Accepts user inputs via HTML form
- Converts inputs into NumPy arrays
- Loads trained model
- Displays prediction result

---

## 🚀 CI/CD & Deployment

### Docker
- Application is containerized
- Ensures environment consistency

### Jenkins
- Automates build and deployment pipeline
- Builds Docker image
- Pushes image to Google Container Registry
- Triggers Cloud Run deployment

### Google Cloud Run
- Serverless deployment
- Auto-scaling
- Public endpoint access

---

## 📸 Screenshots

Add screenshots inside a `screenshots/` folder:


Include:
- Flask prediction UI
- MLflow experiments dashboard
- Jenkins CI/CD pipeline
- Cloud Run deployed service

---

## 🌟 What Makes This Project Unique

- Full **end-to-end MLOps lifecycle**
- Cloud-native architecture
- Experiment tracking with MLflow
- CI/CD automation for ML workflows
- Production deployment (not just notebooks)
- Real-world business use case

---

## 📚 Key Learnings

- Building production-ready ML systems
- Cloud-based data ingestion
- Feature engineering and model tuning
- Experiment tracking and reproducibility
- CI/CD for ML projects
- Docker and serverless deployment
- Industry-level MLOps practices

---

## 🔮 Future Enhancements

- Model monitoring and drift detection
- Data versioning using DVC
- REST API integration
- Automated retraining schedules
- Role-based access control

---

## 👤 Author

**Prashanth H V**  
Final Year Data Science Student  
Aspiring Data Scientist | ML Engineer | MLOps Engineer  

🔗 LinkedIn: <your-link>  
🔗 GitHub: <your-link>

---

⭐ If you found this project useful, please give it a star!
