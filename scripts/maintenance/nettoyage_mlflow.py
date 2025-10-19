# scripts/maintenance/nettoyage_mlflow.py
import argparse
import mlflow
from mlflow.tracking import MlflowClient

def keep_topk(experiment_name: str, metric_key: str, k: int, ascending: bool, dry_run: bool):
    """Supprime tous les runs sauf les K meilleurs d'une expérience MLflow donnée."""
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"[⚠️] Expérience '{experiment_name}' introuvable.")
        return

    # Récupération de tous les runs (on évite les filtres non supportés par certains backends)
    try:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            max_results=50000,
        )
    except Exception as e:
        print(f"[ℹ️] search_runs avec filtre 'status=FINISHED' a échoué ({e}). "
              "On récupère tous les runs et on filtrera côté Python.")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=50000,
        )
        # Filtre côté Python : ne garder que les FINISHED
        runs = [r for r in runs if getattr(r.info, "status", None) == "FINISHED"]
    # Certains backends ne permettent pas de filtrer par lifecycle_stage dans la requête.
    # On filtre côté Python pour ne garder que les runs actifs (si attribut présent).
    runs = [r for r in runs if getattr(r.info, "lifecycle_stage", "active") == "active"]

    # Sanity log
    print(f"[{experiment_name}] runs récupérés après filtrage: {len(runs)}")

    # Tri et sélection
    runs_sorted = sorted(
        runs,
        key=lambda r: r.data.metrics.get(metric_key, float('-inf' if not ascending else 'inf')),
        reverse=not ascending
    )
    keep = runs_sorted[:k]
    drop = runs_sorted[k:]

    if len(runs_sorted) == 0:
        print(f"[{experiment_name}] Aucun run trouvé (status=FINISHED).")
        return

    print(f"[{experiment_name}] total={len(runs_sorted)} | garder={len(keep)} | supprimer={len(drop)} (critère: {metric_key})")
    if dry_run:
        print("🟡 Mode simulation activé (aucune suppression).")
        return

    # Suppression
    for r in drop:
        client.delete_run(r.info.run_id)
    print(f"[{experiment_name}] ✅ Suppression terminée ({len(drop)} runs supprimés).")

def main():
    ap = argparse.ArgumentParser(description="Nettoyage sélectif des runs MLflow")
    ap.add_argument("--experiment", required=True, help="Nom exact de l'expérience MLflow")
    ap.add_argument("--metric", required=True, help="Nom de la métrique de tri (ex: val_f1, f1_macro)")
    ap.add_argument("--top_k", type=int, default=3, help="Nombre de meilleurs runs à conserver")
    ap.add_argument("--ascending", action="store_true", help="Tri croissant (par défaut décroissant)")
    ap.add_argument("--dry_run", action="store_true", help="Simulation sans suppression réelle")
    args = ap.parse_args()
    keep_topk(args.experiment, args.metric, args.top_k, args.ascending, args.dry_run)

if __name__ == "__main__":
    main()