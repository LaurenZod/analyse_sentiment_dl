import json
from collections import defaultdict
import os # Importation nécessaire

LOG_FILE = "production_inferences.jsonl"
TRIGGER_FILE = "accuracy_trigger.txt" # Nouveau fichier de sortie pour l'Accuracy

def analyze_production_log(log_path: str):
    """Analyse le fichier de log pour calculer le Prediction Drift et le Quality Drift."""
    
    total_inferences = 0
    label_counts = defaultdict(int)
    total_score = 0
    
    correct_predictions_with_gt = 0
    total_inferences_with_gt = 0
    
    # Lecture du fichier ligne par ligne
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # 1. Calculs pour le Prediction Drift
                    label = data.get("prediction_label")
                    if label:
                        label_counts[label] += 1
                        total_inferences += 1
                        
                    score = data.get("prediction_score")
                    if score is not None:
                        total_score += score
                        
                    # 2. Calculs pour le Quality Drift (si Ground Truth est disponible)
                    ground_truth = data.get("ground_truth")
                    if ground_truth is not None and ground_truth.lower() in ["positive", "negative"]:
                        total_inferences_with_gt += 1
                        if data.get("prediction_label") == ground_truth.lower():
                            correct_predictions_with_gt += 1
                        
                except json.JSONDecodeError:
                    # Gère les lignes mal formées si elles existent
                    continue
                    
    except FileNotFoundError:
        print(f"Erreur: Fichier de log non trouvé à: {log_path}")
        return

    # [CODE D'AFFICHAGE OMIS POUR BREVITÉ, MAIS IL RESTE DANS LE SCRIPT COMPLET]
    # ...

    # Calcul de la précision (Quality Drift)
    print("\n## 📉 3. Model Quality Drift (Précision/Accuracy)")
    accuracy = -1.0 # Valeur par défaut pour l'échec
    
    if total_inferences_with_gt > 10: # Déclencher le ré-entraînement seulement avec assez de données
        accuracy = (correct_predictions_with_gt / total_inferences_with_gt) * 100
        print(f"- Précision (Accuracy) sur vérité terrain: {accuracy:.2f}%")
        print(f"- Basé sur {total_inferences_with_gt} inférences labellisées.")
    else:
        print(f"- Pas assez de vérités terrain (Ground Truth) ({total_inferences_with_gt}/10) pour calculer une précision fiable.")
        accuracy = 100.0 # Assurer qu'il n'y ait pas de ré-entraînement inutile au début

    # --- NOUVEAU BLOC: ÉCRITURE DU FICHIER TRIGGER ---
    try:
        with open(TRIGGER_FILE, "w") as f:
            f.write(str(accuracy))
        print(f"\n[INFO] Accuracy écrite dans {TRIGGER_FILE}.")
    except Exception as e:
        print(f"\n[ERREUR] Impossible d'écrire le fichier trigger: {e}")
    # ------------------------------------------------

# Le reste de la fonction main reste inchangé...
if __name__ == "__main__":
    analyze_production_log(LOG_FILE)
