# Kubernetes deployment

Works with Docker Desktop Kubernetes, Minikube, or any cloud cluster (EKS/GKE/AKS).

## 1. Build the image (model must be trained first)

```bash
python -m src.models.train          # produces models/model.joblib
docker build -t heart-disease-api:latest .
```

Minikube only — load the local image into the cluster:

```bash
minikube image load heart-disease-api:latest
```

## 2. Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods -l app=heart-disease-api
kubectl get svc heart-disease-api
```

## 3. Verify

Docker Desktop (LoadBalancer maps to localhost):

```bash
curl http://localhost/health
curl -X POST http://localhost/predict -H "Content-Type: application/json" \
  -d '{"age":57,"sex":1,"cp":4,"trestbps":140,"chol":241,"fbs":0,"restecg":0,"thalach":123,"exang":1,"oldpeak":0.2,"slope":2,"ca":0,"thal":7}'
```

Minikube:

```bash
minikube service heart-disease-api --url   # prints the reachable URL
```

## 4. Tear down

```bash
kubectl delete -f k8s/service.yaml -f k8s/deployment.yaml
```
