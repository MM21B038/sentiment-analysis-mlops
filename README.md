# YouTube Comment Sentiment Analysis

## Overview
This project performs sentiment analysis on YouTube comments using natural language processing (NLP) and machine learning techniques. The pipeline includes data cleaning, feature engineering, model training, and deployment using **DVC (Data Version Control)** and **GitHub Actions for CI/CD**.

## Features
- Data collection and preprocessing of YouTube comments
- Feature extraction using **Sentence Transformers**
- Model training using **XGBoost**
- Experiment tracking with **MLflow**
- CI/CD pipeline for automated model training and deployment
- Deployment using **FastAPI** for serving predictions

## Project Structure
```
.
├── data
│   ├── raw                # Raw data (YouTube comments CSV)
│   ├── processed          # Cleaned data and extracted features
│
├── models                 # Trained machine learning models
│   
├── src
│   ├── ml
│   │   ├── data_cleaning.py        # Data preprocessing
│   │   ├── feature_engineering.py  # Feature extraction
│   │   ├── train.py                # Model training
│   │   ├── promote_model.py        # Model promotion
│   │   ├── inference.py            # Model inference for API
│   ├── api
│       ├── app.py              # FastAPI application for serving predictions
│
├── mlruns                 # MLflow tracking
├── dvc.yaml               # DVC pipeline configuration
├── requirements.txt       # Python dependencies
├── ci.yml                 # GitHub Actions CI/CD pipeline
├── Dockerfile             # Containerization setup
└── README.md              # Project documentation
```

## CI/CD Workflow
- The **GitHub Actions CI/CD pipeline** is triggered on every `push` or `pull request` to the `main` branch.
- It installs dependencies, pulls DVC data, runs the pipeline, and promotes the trained model.
- The latest model is deployed using **FastAPI** in a Docker container.

## Contributing
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes.
4. Push the branch and submit a pull request.

## License
This project is licensed under the MIT License.


