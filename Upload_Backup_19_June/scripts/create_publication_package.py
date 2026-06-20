import os
import shutil

os.makedirs(
    "Final_Paper_Assets/Tables",
    exist_ok=True
)

os.makedirs(
    "Final_Paper_Assets/Figures",
    exist_ok=True
)

os.makedirs(
    "Final_Paper_Assets/Matrices",
    exist_ok=True
)

files = [

    # =====================
    # TABLES
    # =====================

    (
        "paper_assets/final_paper/tables/final_ranking.csv",
        "Final_Paper_Assets/Tables"
    ),

    (
        "paper_assets/final_paper/tables/domain_results_named.csv",
        "Final_Paper_Assets/Tables"
    ),

    (
        "paper_assets/final_paper/tables/ablation_study.csv",
        "Final_Paper_Assets/Tables"
    ),

    (
        "paper_assets/final_paper/tables/meta_feature_importance.csv",
        "Final_Paper_Assets/Tables"
    ),

    (
        "paper_assets/final_paper/tables/ensemble4_metrics.csv",
        "Final_Paper_Assets/Tables"
    ),

    # =====================
    # FIGURES
    # =====================

    (
        "paper_assets/final_paper/plots/final_model_comparison.png",
        "Final_Paper_Assets/Figures"
    ),

    (
        "paper_assets/final_paper/plots/meta_feature_importance.png",
        "Final_Paper_Assets/Figures"
    ),

    (
        "paper_assets/final_paper/roc/ensemble4_roc.png",
        "Final_Paper_Assets/Figures"
    ),

    (
        "paper_assets/final_paper/confusion_matrices/ensemble4_cm.png",
        "Final_Paper_Assets/Figures"
    ),

    # =====================
    # MATRICES
    # =====================

    (
        "paper_assets/final_paper/confusion_matrices/ensemble4_cm.csv",
        "Final_Paper_Assets/Matrices"
    )
]

for src, dst in files:

    if os.path.exists(src):

        shutil.copy(
            src,
            dst
        )

        print("Copied:", src)

    else:

        print("Missing:", src)

print("\nPublication Package Ready")
