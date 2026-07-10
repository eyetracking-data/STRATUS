from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, path: str | Path, ylabel: str | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(kind="bar", x=x, y=y, ax=ax, legend=False)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel or y)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_example_plot(df: pd.DataFrame, path: str | Path, title: str = "Example stream"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["time_s"], df["x_clean"], label="clean x")
    ax.plot(df["time_s"], df["x"], label="output/degraded x", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gaze x")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



def save_grouped_bar(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    title: str,
    path: str | Path,
    ylabel: str,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot(index=index, columns=columns, values=values)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



def save_tradeoff_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    label: str,
    title: str,
    path: str | Path,
    xlabel: str,
    ylabel: str,
):
    """Save a labeled scatter plot for reconstruction/decision trade-offs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(df[x], df[y], s=55)
    for _, row in df.iterrows():
        ax.annotate(
            str(row[label]),
            (row[x], row[y]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
