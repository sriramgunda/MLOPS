flowchart TB

%% =====================
%% DEV & DATA STAGE
%% =====================
A[Raw Data<br/>Heart Disease Dataset]
B[Data Preprocessing<br/>Cleaning & Encoding]
C[Feature Engineering]

A --> B --> C

%% =====================
%% TRAINING & TRACKING
%% =====================
D[Model Training<br/>Logistic Regression / Random Forest]
E[Model Evaluation<br/>Metrics: Accuracy, ROC-AUC]
F[MLflow Tracking<br/>Params • Metrics • Artifacts]

C --> D --> E
D --> F
E --> F

%% =====================
%% MODEL ARTIFACT
%% =====================
G[Trained Model Artifact<br/>Pickle / MLflow Model]

F --> G

%% =====================
%% CI / CD PIPELINE
%% =====================
H[CI Pipeline<br/>Lint • Unit Tests • Build]
I[Docker Image Build]
J[GitHub Container Registry]

G --> H
H --> I --> J

%% =====================
%% DEPLOYMENT
%% =====================
K[Kubernetes Cluster]
L[Model Serving API<br/>FastAPI / Flask]

J --> K
K --> L

%% =====================
%% MONITORING
%% =====================
M[Prometheus<br/>Metrics Collection]
N[Grafana<br/>Dashboards & Alerts]

L --> M --> N
