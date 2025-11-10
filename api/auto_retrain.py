import os
import sys
import time
import subprocess

# --- Configuration du Ré-entraînement ---
ACCURACY_TRIGGER_PATH = "accuracy_trigger.txt"
THRESHOLD = 85.0 # Pourcentage d'Accuracy en dessous duquel nous ré-entraînons
TRAINING_COMMAND = "mlflow run . -e train" # Remplacez par votre vraie commande si elle existe

def check_and_retrain():
    """Vérifie l'Accuracy et déclenche le ré-entraînement si elle est trop basse."""
    
    # 1. Lire le score d'Accuracy
    try:
        with open(ACCURACY_TRIGGER_PATH, "r") as f:
            accuracy = float(f.read().strip())
    except FileNotFoundError:
        print(f"[ALERTE] Fichier trigger {ACCURACY_TRIGGER_PATH} non trouvé. Arrêt.")
        return
    except ValueError:
        print(f"[ALERTE] Contenu du fichier trigger invalide. Arrêt.")
        return

    print(f"\n--- Vérification du Déclenchement de Ré-entraînement ---")
    print(f"Accuracy de production actuelle: {accuracy:.2f}%")
    print(f"Seuil de ré-entraînement: {THRESHOLD:.2f}%")

    # 2. Déclencher le ré-entraînement si le seuil est franchi
    if accuracy < THRESHOLD:
        print("\n🚨 ALERTE : CHUTE DE PERFORMANCE DÉTECTÉE !")
        print(f"Déclenchement du pipeline de ré-entraînement avec la commande: {TRAINING_COMMAND}")

        # --- DÉCLENCHEMENT RÉEL (DÉCOMMENTER POUR ACTIVER) ---
        # try:
        #     # Exécuter la commande d'entraînement (doit être non-bloquante ou bien gérée)
        #     subprocess.run(TRAINING_COMMAND, shell=True, check=True)
        #     print("✅ COMMANDE DE RÉ-ENTRAÎNEMENT EXÉCUTÉE AVEC SUCCÈS.")
        #     # Note: La promotion du modèle devrait être gérée dans le pipeline d'entraînement
        # except subprocess.CalledProcessError as e:
        #     print(f"❌ Échec de l'exécution de la commande de ré-entraînement: {e}")
        # except Exception as e:
        #     print(f"❌ ERREUR LORS DU DÉCLENCHEMENT: {e}")
        
        print("\n[NOTE] La commande de ré-entraînement est actuellement DÉSACTIVÉE (commentée) pour la sécurité.")
    else:
        print("✅ Performance au-dessus du seuil. Aucun ré-entraînement nécessaire.")

if __name__ == "__main__":
    check_and_retrain()
