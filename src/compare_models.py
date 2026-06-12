from pathlib import Path
import pandas as pd


def relative_scale_higher_better(series: pd.Series) -> pd.Series:
    best = series.max()
    if best == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return series / best


def relative_scale_lower_better(series: pd.Series) -> pd.Series:
    best = series.min()
    if best == 0:
        return pd.Series([1.0] * len(series), index=series.index)
    return best / series
    
def select_best_model(
    csv_path: str | Path,
    accuracy_weight: float = 0.7,
    emissions_weight: float = 0.2,
    runtime_weight: float = 0.1,
):
    df = pd.read_csv(csv_path)
    
    df["accuracy_score"] = relative_scale_higher_better(df["accuracy"])
    df["emissions_score"] = relative_scale_lower_better(df["emissions_kgco2e"])
    df["runtime_score"] = relative_scale_lower_better(df["runtime_s"])

    df["score"] = (
        accuracy_weight * df["accuracy_score"]
        + emissions_weight * df["emissions_score"]
        + runtime_weight * df["runtime_score"]
        )
        
    best = df.sort_values(by="score",ascending=False).iloc[0]
    return best
    
if __name__ == "__main__":
    best = select_best_model("results/model_comparison.csv")
    print(best[["model", "accuracy", "emissions_kgco2e", "runtime_s", "score"]])
