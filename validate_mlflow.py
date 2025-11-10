# validate_mlflow.py (À placer dans votre répertoire)
import os
import mlflow

# Récupère l'URI du secret GitHub
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")

if not MLFLOW_URI:
    print("Erreur: MLFLOW_TRACKING_URI non défini.")
    exit(1)

try:
    # Tente de se connecter
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    # Tente d'accéder au client (ceci force la connexion et l'authentification)
    client = mlflow.tracking.MlflowClient()
    
    # Tente d'obtenir la liste des expériences comme test de succès
    client.list_experiments()
    
    print(f"✅ Connexion MLflow réussie à {MLFLOW_URI}")
    exit(0) # Succès
    
except Exception as e:
    print(f"❌ Échec de la connexion ou de l'authentification à MLflow: {e}")
    exit(1) # Échec du Job
