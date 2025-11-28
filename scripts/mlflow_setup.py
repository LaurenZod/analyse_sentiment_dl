"""
mlflow_setup.py
Centralise la config MLflow pour tous les scripts d'entraînement.

- Utilise la variable d'environnement MLFLOW_TRACKING_URI si définie
- Sinon, fallback en local sur file:./mlruns
- Crée / sélectionne l'experiment demandé
"""

import os
import mlflow


def init_mlflow(experiment_name: str) -> None:
    """
    Configure le tracking MLflow et sélectionne (ou crée) un experiment.
    """
    # 1) URI du serveur MLflow (EC2 ou local)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # 2) Sélectionne ou crée l'experiment
    mlflow.set_experiment(experiment_name)