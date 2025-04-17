import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
# 1) helper to parse quarter strings
# ────────────────────────────────────────────────────────────────
def parse_quarter_label(q_label):
    # "Q3 2021" → (2021, 3)
    parts = q_label.split()
    return int(parts[1]), int(parts[0][1])

def quarter_to_numeric(year, qtr):
    # for plotting on a single axis
    return year + qtr / 10.0

# ────────────────────────────────────────────────────────────────
# 2) bucket revision‐dates into 3 / 2 / 1 month before quarter‐end
# ────────────────────────────────────────────────────────────────
def assign_rev_bucket(df, quarter_ends):
    # quarter_ends: dict mapping "Q3 2021" → Timestamp('2021‑09‑30')
    df = df.copy()
    # strip off the leading "Rev " and parse
    df['RevDate'] = pd.to_datetime(df['Revision Time'].str.replace('Rev ', ''))
    df['QuarterEnd'] = df['Quarter'].map(quarter_ends)
    # months difference (rounded)
    df['MonthsBefore'] = ((df['QuarterEnd'] - df['RevDate'])
                           / np.timedelta64(1, 'M')).round().astype(int)
    # map to bucket labels
    df['RevBucket'] = df['MonthsBefore'].map({
        3: 'Rev 3m',
        2: 'Rev 2m',
        1: 'Rev 1m'
    }).fillna('other')
    return df

# ────────────────────────────────────────────────────────────────
# 3) main viz routine
# ────────────────────────────────────────────────────────────────
def visualize_random_companies(
    df,
    metric="Stock Price",
    m=3,
    n=3,
    random_state=42
):
    """
    For m random companies (of a chosen metric) and n analysts each,
    overlays:
      - actual Q0 line in black
      - all forecasts colored by Rev‐bucket, shaped by Horizon
    """
    rng = np.random.default_rng(random_state)
    df_m = df[df["Estimate Type"] == metric].copy()
    if df_m.empty:
        print(f"No rows for metric={metric}")
        return

    # build a map Quarter→quarter_end date
    # assumes df.quarter strings match your quarter_ends in main script
    unique_q = sorted(df_m["Quarter"].unique(),
                      key=lambda l: parse_quarter_label(l))
    # you must have imported quarter_ends array in your notebook
    # e.g. quarter_ends = pd.date_range(...).to_series().groupby(...).last()
    # here we assume you passed it in or rebuild it:
    # quarter_ends_map = { f"Q{d.quarter} {d.year}": d for d in quarter_ends }
    # for brevity just re‑derive from df_m Real Value dates:
    # — ***in your notebook before calling this, define***
    #      quarter_ends_map = {...} 
    # and then monkey‑patch into this function if you like.
    # For now: grab all real‐value rows
    real_lookup = {}
    for _, row in df_m.dropna(subset=["Real Value"]).iterrows():
        real_lookup[(row["Quarter"], row["Company"], row["Horizon"])] = row["Real Value"]

    # simplify revision‐times to 3 buckets
    # (you will need to pass in the actual quarter_ends map from your notebook)
    # here I assume you monkey‑patched quarter_ends_map into the module:
    try:
        qmap = visualize_random_companies._quarter_ends_map
    except AttributeError:
        raise RuntimeError(
            "Please attach a dict quarter_ends_map to visualize_random_companies"
        )
    df_m = assign_rev_bucket(df_m, qmap)

    # pick random companies
    comps = rng.choice(df_m["Company"].unique(),
                       size=min(m, df_m["Company"].nunique()),
                       replace=False)

    # set up color & marker maps
    color_map  = {'Rev 3m':'C0','Rev 2m':'C1','Rev 1m':'C2'}
    marker_map = {'Q0':'o','Q1':'^','Q2':'s'}

    for c in comps:
        df_c = df_m[df_m["Company"] == c].copy()
        # parse & numeric quarter for plotting
        df_c[["Y","Q"]] = df_c["Quarter"].apply(parse_quarter_label).tolist()
        df_c["QuarterNum"]       = df_c.apply(lambda r: quarter_to_numeric(r.Y,r.Q), axis=1)

        # actual Q0 line
        real = (df_c[df_c["Horizon"]=="Q0"]
                .drop_duplicates("QuarterNum")
                .sort_values("QuarterNum"))
        plt.figure(figsize=(10,5))
        plt.plot(real["QuarterNum"],
                 real["Real Value"],
                 '-k', lw=2, marker='o',
                 label="Actual Q0")

        # now overlay forecasts
        for (rev, hor, analyst), grp in df_c.groupby(["RevBucket","Horizon","Analyst"]):
            if rev not in color_map or hor not in marker_map:
                continue
            # for each target quarter only plot the mean predicted value
            preds = (grp.groupby("QuarterNum")
                     ["Predicted Value"]
                     .mean()
                     .reset_index())
            plt.scatter(preds["QuarterNum"],
                        preds["Predicted Value"],
                        c=color_map[rev],
                        marker=marker_map[hor],
                        alpha=0.7,
                        label=f"{rev} {hor}"
                              if analyst==grp["Analyst"].iloc[0] and hor=='Q0'
                              else "")

        plt.title(f"{c} – {metric}")
        plt.xlabel("Target Quarter (year.quarter)")
        plt.ylabel(metric)
        # reduce legend to unique entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_lbl = dict(zip(labels, handles))
        plt.legend(by_lbl.values(), by_lbl.keys(), bbox_to_anchor=(1.02,1), loc="upper left")
        plt.tight_layout()
        plt.show()

# store your quarter→end‑date map here before calling:
# visualize_random_companies._quarter_ends_map = { f"Q{d.quarter} {d.year}": d for d in quarter_ends }

# ────────────────────────────────────────────────────────────────
# 4) simple coverage histogram
# ────────────────────────────────────────────────────────────────
def plot_analyst_coverage_hist(df, col_company="Company", col_analyst="Analyst",
                               bins=30, title="Analysts per Company"):
    cover = df.groupby(col_company)[col_analyst].nunique()
    plt.figure(figsize=(8,5))
    plt.hist(cover, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel("Number of Unique Analysts")
    plt.ylabel("Count of Companies")
    plt.tight_layout()
    plt.show()
    return cover