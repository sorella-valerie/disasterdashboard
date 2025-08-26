"""
Disaster Law Dashboard
A Streamlit dashboard for analyzing emergency management data across US regions
with a focus on Equity Initiatives, Mutual Aid, Mitigation Planning,
Local Emergency Powers, and Vulnerable Populations.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Disaster Law Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f4788; text-align: center; margin-bottom: 1.5rem; }
    .chart-card { background-color: #f7f8fa; padding: 1rem; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); height: 520px; display: flex; flex-direction: column; }
    .chart-card h3 { margin: 0 0 0.5rem 0; font-size: 1.05rem; font-weight: 600; color: #222; }
    .chart-wrap { flex: 1; min-height: 0; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.05rem; }
    div[data-testid="column"] > div > div > div > div[data-testid="stCheckbox"] { margin-top: 30px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Disaster Law Dashboard</h1>', unsafe_allow_html=True)

# Palette
PALETTE = ["#0d1b2a", "#1b263b", "#415a77", "#778da9", "#e0e1dd", "#2a9d8f", "#457b9d"]

# -----------------------
# Data loading
# -----------------------
@st.cache_data
def load_data():
    """
    Load Excel files from repo root or data/ by relaxed patterns.
    Returns: dict[label->df], dict[label->path], list[missing labels]
    """
    patterns = {
        'AK-HI': r'AKHI.*\.xlsx',
        'Appalachia-Central': r'AppalachiaCentral.*\.xlsx',
        'CA-WA-OR': r'CAWAOR.*\.xlsx',
        'Midwest-State': r'MidwestState.*\.xlsx',
        'Mountain-West': r'MTNWest.*\.xlsx',
        'Northeast': r'Northeast.*\.xlsx',
        'South-MidAtlantic': r'South[ _]MidAtlantic.*\.xlsx',
        'Southwest': r'(?:^SW|Southwest).*Emergency.*\.xlsx'
    }
    search_dirs = [Path('.'), Path('data')]
    dataframes, missing, loaded_files = {}, [], {}

    def find_match(pat: str):
        rx = re.compile(pat, re.IGNORECASE)
        for d in search_dirs:
            if d.exists():
                for p in d.glob('*.xlsx'):
                    if rx.fullmatch(p.name) or rx.search(p.name):
                        return p
        return None

    for label, pat in patterns.items():
        path = find_match(pat)
        if path and path.exists():
            try:
                df = pd.read_excel(path, engine='openpyxl')
                df.columns = df.columns.str.strip()
                if 'State/Territory' in df.columns:
                    df.rename(columns={'State/Territory': 'State'}, inplace=True)
                dataframes[label] = df
                loaded_files[label] = str(path)
            except Exception as e:
                st.warning(f"Could not load {path.name}: {e}")
        else:
            missing.append(label)

    return dataframes, loaded_files, missing


@st.cache_data
def prepare_combined_data(dataframes: dict) -> pd.DataFrame:
    """Combine selected files and normalize key columns."""
    emergency_files = ['CA-WA-OR', 'Southwest', 'Midwest-State']
    combined = []
    for key in emergency_files:
        if key in dataframes:
            df = dataframes[key].copy()
            df['Region'] = key.replace('-', ' ')
            combined.append(df)
    if not combined:
        return pd.DataFrame()

    combined_df = pd.concat(combined, ignore_index=True)
    column_mapping = {
        'Explicit Vulnerable Population Provisions': 'Vulnerable Populations',
        'Vulnerable Population Provisions': 'Vulnerable Populations',
        'Explicit Vulnerable Populations Protections': 'Vulnerable Populations',
        'Equity/Health Disparities Focus': 'Equity Initiatives',
        'State Emergency Declaration': 'Emergency Declaration'
    }
    combined_df.rename(columns=column_mapping, inplace=True)
    return combined_df

# -----------------------
# Helpers
# -----------------------
def _present_value(v) -> bool:
    if pd.isna(v):
        return False
    return str(v).strip().lower() not in ("", "none", "nan")

def _presence_mask(df: pd.DataFrame, col_name: str) -> pd.Series:
    if col_name not in df.columns:
        return pd.Series(False, index=df.index)
    block = df[col_name] if isinstance(df[col_name], pd.DataFrame) else df[col_name].to_frame()
    return block.applymap(_present_value).any(axis=1).astype(bool).reindex(df.index, fill_value=False)

# -----------------------
# Main app
# -----------------------
def main():
    dataframes, loaded_files, missing = load_data()
    combined_df = prepare_combined_data(dataframes)

    with st.expander("Loaded datasets"):
        if loaded_files:
            st.write(loaded_files)
        if missing:
            st.info(f"Missing datasets: {', '.join(missing)}")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select View:", ["Summary", "Category Deep Dive", "State Comparison"])

    target_columns = ['Equity Initiatives', 'Mutual Aid', 'Mitigation Planning', 'Local Emergency Powers', 'Vulnerable Populations']

    # ---------------- Summary ----------------
    if page == "Summary":
        st.subheader("Summary")

        # Coverage by category
        coverage_df = pd.DataFrame(columns=["Category", "Coverage %", "States Covered", "Total States"])
        if not combined_df.empty:
            rows = []
            total_states = combined_df['State'].dropna().nunique() if 'State' in combined_df.columns else 0
            for col in target_columns:
                if col in combined_df.columns:
                    mask = _presence_mask(combined_df, col)
                    covered = combined_df.loc[mask, 'State'].dropna().nunique() if total_states else 0
                    pct = (covered / total_states * 100) if total_states else 0.0
                    rows.append({"Category": col, "Coverage %": pct, "States Covered": covered, "Total States": total_states})
            if rows:
                coverage_df = pd.DataFrame(rows)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Coverage by Category")
            if not coverage_df.empty:
                fig_cov = px.bar(
                    coverage_df, x="Category", y="Coverage %", text="Coverage %",
                    color="Category", color_discrete_sequence=PALETTE
                )
                fig_cov.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
                fig_cov.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=360, showlegend=False,
                                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_cov, use_container_width=True, config={"displayModeBar": False}, key="summary_cov_by_cat")
            else:
                st.info("No coverage data available.")

        with c2:
            st.subheader("State Coverage Heatmap")
            if not combined_df.empty and 'State' in combined_df.columns:
                # Build a binary coverage table by state and category
                states = sorted(combined_df['State'].dropna().unique().tolist())
                mat = pd.DataFrame({'State': states})
                for col in target_columns:
                    if col in combined_df.columns:
                        present_by_state = combined_df.groupby('State')[col].apply(lambda s: any(_present_value(v) for v in s)).reindex(states, fill_value=False)
                        mat[col] = present_by_state.astype(int).values
                pivot = mat.set_index('State').T  # categories x states
                fig_heat = px.imshow(pivot, labels=dict(x="State", y="Category", color="Coverage"),
                                     color_continuous_scale=["#0d1b2a", "#1b263b", "#2a9d8f", "#52b788"])
                fig_heat.update_layout(width=800, height=360, xaxis=dict(side="bottom"),
                                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False}, key="summary_heatmap")
            else:
                st.info("No heatmap data available.")

        # Average by region
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Average Coverage by Region")
            if not combined_df.empty and {'Region', 'State'}.issubset(combined_df.columns):
                region_rows = []
                for region, g in combined_df.groupby('Region', dropna=False):
                    n_states = g['State'].dropna().nunique()
                    shares = []
                    for col in target_columns:
                        if col in g.columns and n_states:
                            mask = _presence_mask(g, col)
                            covered = g.loc[mask, 'State'].dropna().nunique()
                            shares.append(covered / n_states)
                    avg_share = np.mean(shares) if shares else 0.0
                    region_rows.append({'Region': region, 'Share': avg_share})
                if region_rows:
                    df_region = pd.DataFrame(region_rows)
                    fig_region = px.bar(df_region, x="Region", y="Share", text="Share", color="Region",
                                        color_discrete_sequence=PALETTE)
                    fig_region.update_traces(texttemplate="%{text:.0%}", textposition="outside", cliponaxis=False)
                    fig_region.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=360, yaxis_tickformat=".0%",
                                             showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_region, use_container_width=True, config={"displayModeBar": False}, key="summary_region_avg")
                else:
                    st.info("No region coverage data available.")
            else:
                st.info("No region coverage data available.")

        # Top states
        with c4:
            st.subheader("Top States by Categories Covered")
            if not combined_df.empty and 'State' in combined_df.columns:
                parts = []
                for col in target_columns:
                    if col in combined_df.columns:
                        mask = _presence_mask(combined_df, col)
                        tmp = pd.DataFrame({'State': combined_df['State'], col: mask.astype(int)})
                        tmp = tmp.groupby('State', as_index=False)[col].max().set_index('State')
                        parts.append(tmp)
                if parts:
                    joined = pd.concat(parts, axis=1).fillna(0).astype(int)
                    joined['Total Categories'] = joined.sum(axis=1)
                    top_df = joined['Total Categories'].sort_values(ascending=False).head(15).reset_index()
                    top_df.columns = ['State', 'Total Categories']
                    fig_top = px.bar(
                        top_df.sort_values("Total Categories"),
                        x="Total Categories", y="State",
                        orientation="h", text="Total Categories",
                        color="Total Categories",
                        color_continuous_scale=[PALETTE[0], PALETTE[2], PALETTE[5]]
                    )
                    fig_top.update_traces(textposition="outside")
                    fig_top.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=360, showlegend=False,
                                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False}, key="summary_top_states")
                else:
                    st.info("No state coverage ranking available.")
            else:
                st.info("No state coverage ranking available.")

    # --------------- Category Deep Dive ---------------
    elif page == "Category Deep Dive":
        st.header("Category Deep Dive")

        selected = st.multiselect("Select Categories:", target_columns, default=target_columns)

        # Gather datasets that have any selected category
        dfs_with_category = []
        for name, df in dataframes.items():
            cols_str = [str(c) for c in df.columns]
            if any(cat in c for cat in selected for c in cols_str):
                dfs_with_category.append((name, df))

        if not dfs_with_category:
            st.warning("No datasets contain the selected categories.")
        else:
            st.write(f"Found in {len(dfs_with_category)} datasets:")
            tabs = st.tabs([name for name, _ in dfs_with_category])

            for tab_idx, (tab, (name, df)) in enumerate(zip(tabs, dfs_with_category)):
                with tab:
                    # Find actual matching columns
                    present_cols = []
                    for cat in selected:
                        for col in df.columns:
                            if cat in str(col):
                                present_cols.append((cat, col))

                    if not present_cols:
                        st.warning("No matching columns found.")
                        continue

                    if 'State' in df.columns:
                        for col_idx, (cat, actual_col) in enumerate(present_cols):
                            block_key = f"{tab_idx}_{col_idx}"

                            st.subheader(f"{name} • {cat}")

                            non_empty = df[df[actual_col].apply(_present_value)]
                            total_rows = len(df)
                            covered_rows = len(non_empty)
                            pct = (covered_rows / total_rows * 100) if total_rows else 0.0

                            cA, cB = st.columns([1, 2])
                            with cA:
                                st.metric("Coverage", f"{pct:.1f}%")
                                st.write(f"States with {cat}: {covered_rows}/{total_rows}")

                            with cB:
                                if covered_rows:
                                    mini_data = pd.DataFrame({
                                        "Status": ["Covered", "Not Covered"],
                                        "Count": [covered_rows, total_rows - covered_rows]
                                    })
                                    fig_mini = px.bar(
                                        mini_data, x="Status", y="Count",
                                        color="Status",
                                        color_discrete_map={"Covered": PALETTE[5], "Not Covered": PALETTE[1]}
                                    )
                                    fig_mini.update_layout(height=200, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
                                    st.plotly_chart(
                                        fig_mini,
                                        use_container_width=True,
                                        config={"displayModeBar": False},
                                        key=f"deepdive_chart_{name}_{cat}_{block_key}"
                                    )

                            if covered_rows:
                                with st.expander(f"View data for {covered_rows} states"):
                                    show_cols = ['State', actual_col] if 'State' in non_empty.columns else [actual_col]
                                    st.dataframe(
                                        non_empty[show_cols],
                                        use_container_width=True,
                                        key=f"deepdive_table_{name}_{cat}_{block_key}"
                                    )
                    else:
                        disp_cols = [c for _, c in present_cols]
                        st.dataframe(
                            df[disp_cols].head(),
                            use_container_width=True,
                            key=f"deepdive_no_state_{name}"
                        )

    # --------------- State Comparison ---------------
    elif page == "State Comparison":
        st.header("State Comparison")

        all_states = sorted(set().union(
            *(df['State'].dropna().unique() for df in dataframes.values() if 'State' in df.columns)
        ))
        selected_states = st.multiselect("Select states to compare:", all_states, default=all_states[:3])

        if selected_states:
            comp_rows = []
            for s in selected_states:
                row = {'State': s}
                for col in target_columns:
                    present = False
                    for df in dataframes.values():
                        if 'State' in df.columns and s in set(df['State'].dropna()):
                            series = df.loc[df['State'] == s, col] if col in df.columns else pd.Series([], dtype=object)
                            if series.apply(_present_value).any():
                                present = True
                                break
                    row[col] = present
                comp_rows.append(row)

            comp_df = pd.DataFrame(comp_rows)

            # Radar
            fig_rad = go.Figure()
            for idx, row in comp_df.iterrows():
                vals = [1 if row[c] else 0 for c in target_columns]
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals, theta=target_columns, fill='toself', name=row['State'], opacity=0.6
                ))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor='lightgray'),
                           bgcolor='rgba(0,0,0,0)'),
                showlegend=True, height=420,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_rad, use_container_width=True, key="state_comp_radar")

            # Matrix
            comp_disp = comp_df.copy()
            for c in target_columns:
                comp_disp[c] = comp_disp[c].map({True: "✓", False: "✗"})
            st.dataframe(comp_disp, use_container_width=True, hide_index=True, height=400, key="state_comp_table")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        Disaster Law Dashboard<br>
        Focus Areas: Equity Initiatives • Mutual Aid • Mitigation Planning • Local Emergency Powers • Vulnerable Populations
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
