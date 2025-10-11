# mlflow_setup.py
import mlflow

def init_mlflow(experiment_name: str = "experiments"):
    """
    Initialise MLflow pour stocker localement dans ./mlruns
    et sélectionne (ou crée) l'expérience.
    """
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)