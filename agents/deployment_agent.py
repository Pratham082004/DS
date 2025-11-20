"""
agents/deployment_agent.py

DeploymentAgent (Production-ready)
---------------------------------
Responsibilities:
- Register trained model artifacts to a local registry (filesystem or S3)
- Generate a production-ready REST API scaffold (FastAPI) that loads the model and exposes /predict
- Generate a Dockerfile and a basic Kubernetes Deployment+Service YAML for containerized deployment
- Produce health-check scripts and a CI/CD checklist file

Outputs:
- model registry entry (data/registry/<model_id>.json) and optional copy of model .pkl
- deployment scaffold (deploy/<model_name>/{app.py, Dockerfile, k8s/*.yaml})
- metadata JSON describing endpoint, ports, and health-checks
"""

import os
import json
import shutil
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DeploymentAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {
            "registry_dir": "data/registry",
            "deploy_dir": "deploy",
            "python_base_image": "python:3.11-slim",
            "service_port": 8080,
            "model_mount_dir": "/app/models",
        }
        if config:
            cfg.update(config)
        self.cfg = cfg
        os.makedirs(self.cfg["registry_dir"], exist_ok=True)
        os.makedirs(self.cfg["deploy_dir"], exist_ok=True)

    # ---------------- helpers ----------------
    def _save_json(self, obj: Any, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)

    # ---------------- model registry ----------------
    def register_model(self, model_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Registers a model file into the local registry.
        Copies the .pkl to registry with a generated id and stores metadata JSON.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model_name = Path(model_path).stem
        model_id = f"{model_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
        target_dir = Path(self.cfg["registry_dir"]) / model_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # copy model
        target_model_path = target_dir / Path(model_path).name
        shutil.copyfile(model_path, str(target_model_path))

        # create metadata
        meta = {
            "model_id": model_id,
            "model_name": model_name,
            "registered_at": datetime.utcnow().isoformat() + "Z",
            "model_path": str(target_model_path),
            "metadata": metadata or {}
        }
        meta_path = target_dir / "metadata.json"
        self._save_json(meta, str(meta_path))
        logging.info(f"Model registered: {model_id}")
        return meta

    # ---------------- REST scaffold generator ----------------
    def generate_rest_scaffold(self, model_registry_meta: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a minimal FastAPI scaffold that loads the model and exposes /predict.
        Creates Dockerfile and a simple requirements.txt for deployment.
        """
        out_root = Path(output_dir or (Path(self.cfg["deploy_dir"]) / model_registry_meta["model_id"]))
        out_root.mkdir(parents=True, exist_ok=True)

        model_rel_path = Path(model_registry_meta["model_path"]).name
        # copy model into scaffold folder (so Docker build can add it)
        shutil.copyfile(model_registry_meta["model_path"], str(out_root / model_rel_path))

        # app.py content (FastAPI)
        app_py = f'''"""
Auto-generated FastAPI server for model {model_registry_meta["model_name"]}
Run: uvicorn app:app --host 0.0.0.0 --port {self.cfg['service_port']}
"""
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List

app = FastAPI(title="Model API - {model_registry_meta['model_name']}")

# Load model (adjust path if needed)
model = joblib.load("{model_rel_path}")

class PredictPayload(BaseModel):
    inputs: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {{"status": "ok"}}

@app.post("/predict")
def predict(payload: PredictPayload):
    try:
        # expect inputs: list of feature dicts
        import pandas as pd
        X = pd.DataFrame(payload.inputs)
        preds = model.predict(X)
        return {{"predictions": preds.tolist()}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={self.cfg['service_port']})
'''

        (out_root / "app.py").write_text(app_py, encoding="utf-8")

        # requirements.txt
        requirements = "\n".join(["fastapi", "uvicorn[standard]", "scikit-learn", "pandas", "joblib"])
        (out_root / "requirements.txt").write_text(requirements, encoding="utf-8")

        # Dockerfile
        dockerfile = f'''
FROM {self.cfg["python_base_image"]}
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE {self.cfg["service_port"]}
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "{self.cfg['service_port']}"]
'''
        (out_root / "Dockerfile").write_text(dockerfile.strip(), encoding="utf-8")

        # k8s manifest (deployment + service)
        k8s_dir = out_root / "k8s"
        k8s_dir.mkdir(exist_ok=True)
        deployment_yaml = f'''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {model_registry_meta["model_id"]}-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {model_registry_meta["model_id"]}
  template:
    metadata:
      labels:
        app: {model_registry_meta["model_id"]}
    spec:
      containers:
      - name: model-server
        image: REPLACE_WITH_YOUR_IMAGE
        ports:
        - containerPort: {self.cfg["service_port"]}
        env:
        - name: MODEL_ID
          value: "{model_registry_meta['model_id']}"
---
apiVersion: v1
kind: Service
metadata:
  name: {model_registry_meta["model_id"]}-svc
spec:
  selector:
    app: {model_registry_meta["model_id"]}
  ports:
  - protocol: TCP
    port: {self.cfg["service_port"]}
    targetPort: {self.cfg["service_port"]}
'''
        (k8s_dir / "deployment.yaml").write_text(deployment_yaml.strip(), encoding="utf-8")

        logging.info(f"Generated scaffold at {str(out_root)}")
        return {"scaffold_path": str(out_root), "dockerfile": str(out_root / "Dockerfile"), "k8s_manifest": str(k8s_dir / "deployment.yaml")}

    # ---------------- Health check & CI hints ----------------
    def health_check_script(self, output_dir: Optional[str] = None) -> str:
        out_root = Path(output_dir or self.cfg["deploy_dir"])
        out_root.mkdir(parents=True, exist_ok=True)
        script_path = out_root / "health_check.sh"
        content = f'''#!/usr/bin/env bash
# Simple health check script for the model API
HOST=${{1:-http://localhost:{self.cfg['service_port']}}}
echo "Checking health at $HOST/health"
curl -sS "$HOST/health" || (echo "Health check failed" && exit 1)
echo "Health OK"
'''
        script_path.write_text(content, encoding="utf-8")
        os.chmod(script_path, 0o755)
        return str(script_path)

    def ci_cd_checklist(self) -> Dict[str, Any]:
        checklist = {
            "build": ["Dockerfile exists", "requirements.txt exists", "scaffold loads model"],
            "test": ["unit tests (pytest)", "integration test: /predict endpoint", "model contract tests"],
            "security": ["scan dependencies", "use non-root container user", "set resource limits"],
            "deploy": ["update image tag in k8s manifest", "create image registry credentials"]
        }
        return checklist

# ---------------- Local test ----------------
if __name__ == "__main__":
    agent = DeploymentAgent()
    # Example usage: register a model and generate scaffold
    try:
        meta = agent.register_model("data/models/Species_LogisticRegression.pkl", metadata={"source":"local_test"})
        scaffold = agent.generate_rest_scaffold(meta)
        health = agent.health_check_script(scaffold["scaffold_path"])
        print(json.dumps({"meta": meta, "scaffold": scaffold, "health_script": health}, indent=2))
    except Exception as e:
        logging.exception("DeploymentAgent local test failed.")
        raise
