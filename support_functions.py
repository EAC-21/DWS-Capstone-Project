import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
# 1) quarter‐string parsing & numeric conversion
# ────────────────────────────────────────────────────────────────
def parse_quarter_label(q_label):
    parts = q_label.split()
    return int(parts[1]), int(parts[0][1])

def quarter_to_numeric(year, qtr):
    return year + (qtr - 1) * 0.25

# ────────────────────────────────────────────────────────────────
# 2) bucket each Rev‐date into 3m / 2m / 1m before quarter‐end
# ────────────────────────────────────────────────────────────────
def assign_rev_bucket(df, quarter_ends_map):
    df = df.copy()
    df = df[df["Revision Time"].str.startswith("Rev ")].copy()
    df['RevDate'] = pd.to_datetime(df['Revision Time'].str.replace('Rev ', ''), format='%Y-%m-%d')
    df['QuarterEnd'] = df['Quarter'].map(quarter_ends_map)

    # FIXED: Compute months between dates manually (calendar difference)
    df['MonthsBefore'] = (
        (df['QuarterEnd'].dt.year - df['RevDate'].dt.year) * 12 +
        (df['QuarterEnd'].dt.month - df['RevDate'].dt.month)
    )

    bucket_map = {3: 'Rev 3m', 2: 'Rev 2m', 1: 'Rev 1m'}
    df['RevBucket'] = df['MonthsBefore'].map(bucket_map)
    return df[df['RevBucket'].notna()]

# ────────────────────────────────────────────────────────────────
# 3) main visualization routine
# ────────────────────────────────────────────────────────────────
def visualize_random_companies(df, metric="Stock Price", m=3, n=3, random_state=42):
    rng = np.random.default_rng(random_state)
    df_m = df[df["Estimate Type"] == metric].copy()
    if df_m.empty:
        print(f"No rows for metric='{metric}'")
        return

    try:
        qmap = visualize_random_companies._quarter_ends_map
    except AttributeError:
        raise RuntimeError("You must inject a dict visualize_random_companies._quarter_ends_map = { ... }")

    df_m = assign_rev_bucket(df_m, qmap)

    comps = rng.choice(df_m["Company"].unique(), size=min(m, df_m["Company"].nunique()), replace=False)

    color_map = {'Rev 3m': 'C0', 'Rev 2m': 'C1', 'Rev 1m': 'C2'}
    marker_map = {'Q0': 'o', 'Q1': '^', 'Q2': 's'}

    for c in comps:
        df_c = df_m[df_m["Company"] == c].copy()
        df_c[['Year','Qtr']] = df_c['Quarter'].apply(lambda x: pd.Series(parse_quarter_label(x)))
        df_c['QuarterNum'] = df_c.apply(lambda r: quarter_to_numeric(r.Year, r.Qtr), axis=1)

        plt.figure(figsize=(10, 5))

                # grab *only* the one true realized Q0 (Initial) per quarter,
        # then sort by its numeric quarter code, and drop any duplicate dates.
        real = df[
            (df["Company"] == c)
          & (df["Estimate Type"] == metric)
          & (df["Horizon"]       == "Q0")
          & (df["Revision Time"] == "Initial")
        ].copy()

        # parse into numeric quarter
        real[['Year','Qtr']] = real['Quarter'].apply(
            lambda x: pd.Series(parse_quarter_label(x))
        )
        real['QuarterNum'] = real.apply(
            lambda r: quarter_to_numeric(r.Year, r.Qtr),
            axis=1
        )

        # now sort chronologically and drop any stray duplicates
        real = real.sort_values('QuarterNum').drop_duplicates('QuarterNum')


        real[['Year','Qtr']] = real['Quarter'].apply(lambda x: pd.Series(parse_quarter_label(x)))
        real['QuarterNum'] = real.apply(lambda r: quarter_to_numeric(r.Year, r.Qtr), axis=1)

        plt.plot(real["QuarterNum"], real["Real Value"], '-k', lw=2, marker='o', label="Actual Q0")

        for (rev_bucket, horizon), grp in df_c.groupby(["RevBucket", "Horizon"]):
            preds = grp.groupby("QuarterNum")["Predicted Value"].mean().reset_index()
            plt.scatter(preds["QuarterNum"], preds["Predicted Value"],
                        c=color_map[rev_bucket],
                        marker=marker_map[horizon],
                        alpha=0.7,
                        label=f"{rev_bucket} {horizon}")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.title(f"Company: {c} | Metric: {metric}")
        plt.xlabel("Target Quarter (year.quarter)")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.show()

# ────────────────────────────────────────────────────────────────
# 4) simple coverage histogram
# ────────────────────────────────────────────────────────────────
def plot_analyst_coverage_hist(df, col_company="Company", col_analyst="Analyst", bins=30, title="Distribution of Analysts per Company"):
    coverage_counts = df.groupby(col_company)[col_analyst].nunique()
    plt.figure(figsize=(8, 5))
    plt.hist(coverage_counts, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel("Number of Unique Analysts")
    plt.ylabel("Count of Companies")
    plt.tight_layout()
    plt.show()
    return coverage_counts