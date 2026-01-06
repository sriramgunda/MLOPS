### Setup and Install Instructions Summary

#### Prerequisites and Local Environment Setup
- **Python Environment**: 
  - Create a virtual environment: `python -m venv venv`
  - Activate it:
    - Linux/Mac: `source venv/bin/activate`
    - Windows: `venv\Scripts\activate`
  - Install dependencies: `pip install -r requirements.txt`
- **IDE**: Visual Studio Code
- **Dataset**: Heart disease dataset (likely heart.csv)
- **Docker**: Required for containerization
- **Kubernetes**: Minikube for local cluster setup
- **Git**: Client for version control and pushing to GitHub

#### Running the Application Locally
1. Activate the virtual environment (as above).
2. Train the model and make predictions: `python train_automl.py && python src/predict.py`
3. Start MLflow UI: `mlflow ui --port 5000` (access at http://127.0.0.1:5000)
4. Run the main app: `python heart_disease.py`
5. Alternative MLflow command: `python -m mlflow ui`

#### Docker Containerization
- Build the image: `docker build -t heart-disease-api:2.0 .`
- Tag the image: `docker tag heart-disease-api:2.0 heart-disease-api:latest`
- Run the container: `docker run -p 8000:8000 heart-disease-api`

#### Kubernetes Deployment with Minikube
1. Start Minikube: `minikube start`
2. Load the Docker image into Minikube: `minikube image load heart-disease-api:latest`
   - Alternative: `docker save my-app:latest | minikube image load -`
3. Apply Kubernetes manifests:
   - `kubectl apply -f k8s/deployment.yaml`
   - `kubectl apply -f k8s/service.yaml`
4. Check status:
   - `kubectl get deployments`
   - `kubectl get pods`
   - `kubectl get services`
5. Access the service: `minikube service heart-disease-service`
6. For external IP (load balancer): `minikube tunnel`

#### Monitoring Setup (Prometheus and Grafana)
- **Standalone Docker**:
  - Prometheus: `docker run --name prometheus -p 9090:9090 -v C:/ddrive/AIML-MS/courses/sem3/MLOPs/Assignment/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus`
  - Grafana: `docker run -d --name grafana -p 3000:3000 grafana/grafana`
- **Kubernetes with Helm**:
  - Add repo: `helm repo add prometheus-community https://prometheus-community.github.io/helm-charts`
  - Update: `helm repo update`
  - Install: `helm install monitoring prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace`
  - Port forward:
    - Prometheus: `kubectl port-forward svc/monitoring-kube-prometheus-prometheus 9090 -n monitoring`
    - Grafana: `kubectl port-forward svc/monitoring-grafana 3000:80 -n monitoring`
  - Grafana admin password: `TS6bN7QZp1IVaB6KyUG8M9ng6HTGAN4SoljdBZHM` (decoded from secret)
- Check endpoints: `kubectl get endpoints heart-disease-service`

#### Additional Notes
- MLflow backend store: `mlflow ui --backend-store-uri file:C:\ddrive\AIML-MS\courses\sem3\MLOPs\Assignment\mlflow-runs`
- Docker registry secret for GitHub Container Registry (ghcr.io) is pre-configured with specific credentials.
- The project includes automated EDA, model training, prediction, and testing scripts in the src and tests directories.

#### Architecture Diagram

![MLOps Architecture Diagram](https://github.com/sriramgunda/MLOPS/blob/sriram/MLOps-architecture.png?raw=true)

#### Recording Link
https://www.youtube.com/watch?v=SqjqfYa2Kyc