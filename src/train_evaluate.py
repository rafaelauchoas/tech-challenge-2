"""
src/train_evaluate.py
Treinamento e avaliação dos modelos de classificação de qualidade de vinho.
Dataset: WineQT.csv

Uso:
    python src/train_evaluate.py
    python src/train_evaluate.py --data-dir ../data --results-dir ../results
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, classification_report,
)
from sklearn.inspection import permutation_importance

from preprocess import full_preprocess

sns.set_theme(style="whitegrid", palette="muted")
RANDOM_STATE = 42


def build_models() -> dict:
    """
    Três modelos em Pipeline (StandardScaler + classificador).
    class_weight='balanced' compensa o desbalanceamento acentuado do WineQT
    (~86% Baixa/Média vs ~14% Alta qualidade).
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced",
                random_state=RANDOM_STATE
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=4, subsample=0.8,
                random_state=RANDOM_STATE
            )),
        ]),
    }


def cross_validate_models(models: dict, X_train, y_train) -> pd.DataFrame:
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipeline in models.items():
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1
        )
        rows.append({"Modelo": name, "AUC_mean": scores.mean(), "AUC_std": scores.std()})
        print(f"[CV] {name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(rows).set_index("Modelo")


def evaluate_on_test(models: dict, X_train, X_test, y_train, y_test,
                     results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    rows    = []
    trained = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline

        y_pred      = pipeline.predict(X_test)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

        rows.append({
            "Modelo":    name,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "F1-Score":  f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC":   roc_auc_score(y_test, y_pred_prob),
        })

        print(f"\n[test] {name}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Baixa/Média", "Alta"]))

    return pd.DataFrame(rows).set_index("Modelo"), trained

def plot_class_balance(y, results_dir: str):
    counts  = y.value_counts()
    labels  = ["Baixa/Média\n(nota < 7)", "Alta\n(nota ≥ 7)"]
    colors  = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, [counts[0], counts[1]], color=colors, edgecolor="white")
    for bar, count in zip(bars, [counts[0], counts[1]]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8,
                f"{count}\n({count/len(y)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Balanceamento das Classes – WineQT")
    ax.set_ylabel("Quantidade de amostras")
    plt.tight_layout()
    path = os.path.join(results_dir, "balanceamento_classes.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] Salvo: {path}")


def plot_quality_distribution(df_full, results_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    df_full["quality"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color="steelblue", edgecolor="white"
    )
    ax.set_title("Distribuição Original das Notas de Qualidade – WineQT")
    ax.set_xlabel("Nota"); ax.set_ylabel("Contagem")
    plt.tight_layout()
    path = os.path.join(results_dir, "distribuicao_notas_original.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] Salvo: {path}")


def plot_feature_distributions(df_full, feature_cols, results_dir: str):
    n    = len(feature_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        sns.histplot(data=df_full, x=col, hue="high_quality", bins=30,
                     ax=axes[i], palette={0: "#4C72B0", 1: "#DD8452"},
                     alpha=0.7, kde=True)
        axes[i].set_title(col)
        axes[i].legend(title="Alta qualidade", labels=["Não", "Sim"])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribuição das Variáveis por Classe", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(results_dir, "distribuicao_variaveis.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[plot] Salvo: {path}")


def plot_correlation(df_full, feature_cols, results_dir: str):
    corr = df_full[feature_cols + ["high_quality"]].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax, linewidths=0.5)
    ax.set_title("Matriz de Correlação", fontsize=14)
    plt.tight_layout()
    path = os.path.join(results_dir, "correlacao.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] Salvo: {path}")

    print("\nCorrelação com high_quality:")
    print(corr["high_quality"].drop("high_quality").sort_values(ascending=False).to_string())


def plot_boxplots(df_full, feature_cols, results_dir: str):
    n    = len(feature_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        sns.boxplot(data=df_full, x="high_quality", y=col, ax=axes[i],
                    hue="high_quality", palette={0: "#4C72B0", 1: "#DD8452"},
                    legend=False)
        axes[i].set_title(col)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Baixa/Média", "Alta"])
        axes[i].set_xlabel("Qualidade")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Boxplots por Classe de Qualidade", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(results_dir, "boxplots.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[plot] Salvo: {path}")


def plot_confusion_matrices(trained: dict, X_test, y_test, results_dir: str):
    fig, axes = plt.subplots(1, len(trained), figsize=(5 * len(trained), 4))
    if len(trained) == 1:
        axes = [axes]

    for ax, (name, pipeline) in zip(axes, trained.items()):
        cm   = confusion_matrix(y_test, pipeline.predict(X_test))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Baixa/Média", "Alta"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)

    plt.suptitle("Matrizes de Confusão", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(results_dir, "confusion_matrices.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[plot] Salvo: {path}")


def plot_roc_curves(trained: dict, X_test, y_test, results_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, pipeline in trained.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("Curvas ROC – Comparativo entre Modelos")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(results_dir, "roc_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] Salvo: {path}")


def plot_metrics_comparison(results_df: pd.DataFrame, results_dir: str):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x       = np.arange(len(metrics))
    width   = 0.25
    colors  = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, row) in enumerate(results_df.iterrows()):
        ax.bar(x + i * width, [row[m] for m in metrics],
               width, label=name, color=colors[i % len(colors)])

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparativo de Métricas por Modelo")
    ax.legend(); ax.set_ylabel("Score")
    plt.tight_layout()
    path = os.path.join(results_dir, "comparativo_metricas.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] Salvo: {path}")


def plot_feature_importance(trained: dict, results_df: pd.DataFrame,
                             feature_names: list, X_test, y_test, results_dir: str):
    # MDI – Random Forest
    if "Random Forest" in trained:
        rf_clf = trained["Random Forest"].named_steps["clf"]
        imp    = pd.Series(rf_clf.feature_importances_,
                           index=feature_names).sort_values()
        fig, ax = plt.subplots(figsize=(8, 7))
        imp.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title("Importância das Features – Random Forest (MDI)")
        ax.set_xlabel("Importância média")
        plt.tight_layout()
        path = os.path.join(results_dir, "feature_importance_rf.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"[plot] Salvo: {path}")

    best_name     = results_df["ROC-AUC"].idxmax()
    best_pipeline = trained[best_name]

    perm = permutation_importance(
        best_pipeline, X_test, y_test,
        n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    perm_df = pd.DataFrame({
        "Feature": feature_names,
        "mean":    perm.importances_mean,
        "std":     perm.importances_std,
    }).sort_values("mean")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(perm_df["Feature"], perm_df["mean"],
            xerr=perm_df["std"], color="coral", edgecolor="white")
    ax.set_title(f"Permutation Importance – {best_name}")
    ax.set_xlabel("Redução média no AUC ao permutar a feature")
    plt.tight_layout()
    path = os.path.join(results_dir, "permutation_importance.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] Salvo: {path}")

def main(data_dir: str = "../data", results_dir: str = "../results"):
    os.makedirs(results_dir, exist_ok=True)

    # 1. Pré-processamento
    X, y = full_preprocess(data_dir, filename="WineQT.csv")
    feature_names = list(X.columns)

    # Carrega df raw separadamente para plots de EDA (precisa da coluna 'quality' original)
    from preprocess import load_wine_data, create_binary_target
    df_raw  = load_wine_data(data_dir, filename="WineQT.csv")
    df_full = create_binary_target(df_raw)   # tem 'quality' + 'high_quality'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTreino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

    # 2. EDA plots
    print("\n=== Gerando gráficos de EDA ===")
    base_features = [c for c in feature_names if c not in
                     ("acidity_ratio", "sulfur_ratio", "alcohol_density")]
    plot_quality_distribution(df_full, results_dir)
    plot_class_balance(y, results_dir)
    plot_feature_distributions(df_full, base_features, results_dir)
    plot_correlation(df_full, base_features, results_dir)
    plot_boxplots(df_full, base_features, results_dir)

    # 3. Modelos
    models = build_models()

    print("\n=== Cross-Validation (5-Fold StratifiedKFold) ===")
    cross_validate_models(models, X_train, y_train)

    print("\n=== Avaliação no Conjunto de Teste ===")
    results_df, trained = evaluate_on_test(
        models, X_train, X_test, y_train, y_test, results_dir
    )

    # 4. Plots de avaliação
    print("\n=== Gerando gráficos de avaliação ===")
    plot_confusion_matrices(trained, X_test, y_test, results_dir)
    plot_roc_curves(trained, X_test, y_test, results_dir)
    plot_metrics_comparison(results_df, results_dir)
    plot_feature_importance(trained, results_df, feature_names, X_test, y_test, results_dir)

    # 5. Salva métricas
    results_df.to_csv(os.path.join(results_dir, "model_metrics.csv"))
    print("\n=== Métricas finais ===")
    print(results_df.to_string())

    best = results_df["ROC-AUC"].idxmax()
    print(f"\nMelhor modelo: {best} (AUC = {results_df.loc[best, 'ROC-AUC']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wine Quality – Treinamento e Avaliação")
    parser.add_argument("--data-dir",    default="../data",    help="Diretório com WineQT.csv")
    parser.add_argument("--results-dir", default="../results", help="Diretório de saída")
    args = parser.parse_args()
    main(args.data_dir, args.results_dir)