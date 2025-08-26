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

# -----------------------
# Data loading
# -----------------------
@st.cache_data
def load_data():
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
    if pd.isna(v): return False
    return str(v).strip().lower() not in ("", "none", "nan")

def _presence_mask(df: pd.DataFrame, col_name: str) -> pd.Series:
    if col_name not in df.columns:
        return pd.Series(False, index=df.index)
    block = df[col_name] if isinstance(df[col_name], pd.DataFrame) else df[col_name].to_frame()
    return block.applymap(_present_value).any(axis=1).astype(bool)

# -----------------------
# Main app
# -----------------------
PALETTE = ["#0d1b2a", "#1b263b", "#415a77", "#778da9", "#e0e1dd", "#2a9d8f", "#457b9d"]

def main():
    dataframes, loaded_files, missing = load_data()
    combined_df = prepare_combined_data(dataframes)

    with st.expander("Loaded datasets"):
        st.write(loaded_files)
        if missing: st.info(f"Missing: {', '.join(missing)}")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select View:", ["Summary", "Category Deep Dive", "State Comparison"])

    target_columns = ['Equity Initiatives', 'Mutual Aid', 'Mitigation Planning', 'Local Emergency Powers', 'Vulnerable Populations']

    if page == "Summary":
        st.subheader("Summary")
        coverage_data = {cat: {'Coverage Percentage': 0, 'States with Coverage': 0, 'Total States': 0} for cat in target_columns}
        if not combined_df.empty:
            for col in target_columns:
                if col in combined_df.columns:
                    mask = _presence_mask(combined_df, col)
                    total = combined_df['State'].nunique()
                    covered = combined_df.loc[mask, 'State'].nunique()
                    coverage_data[col] = {'Coverage Percentage': (covered / total * 100), 'States with Coverage': covered, 'Total States': total}
        coverage_df = pd.DataFrame([{'Category': c, **v} for c, v in coverage_data.items()])

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Coverage by Category")
            fig = px.bar(coverage_df, x="Category", y="Coverage Percentage", color="Category", text="Coverage Percentage", color_discrete_sequence=PALETTE)
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True, key="summary_cov_by_cat")
        with c2:
            st.subheader("Heatmap")
            if not combined_df.empty:
                heatmap_df = combined_df[['State']].drop_duplicates()
                for col in target_columns:
                    if col in combined_df.columns:
                        heatmap_df[col] = combined_df.groupby("State")[col].transform(lambda x: any(map(_present_value, x)))
                pivot = heatmap_df.set_index("State").astype(int).T
                fig_heatmap = px.imshow(pivot, color_continuous_scale=["#0d1b2a","#1b263b","#2a9d8f","#52b788"])
                st.plotly_chart(fig_heatmap, use_container_width=True, key="summary_heatmap")

    elif page == "Category Deep Dive":
        st.header("Category Deep Dive")
        selected = st.multiselect("Select Categories:", target_columns, default=target_columns)
        for name, df in dataframes.items():
            with st.expander(f"{name}"):
                for col in selected:
                    if col in df.columns:
                        mask = _presence_mask(df, col)
                        covered = df[mask]
                        total = len(df)
                        pct = len(covered) / total * 100 if total else 0
                        st.metric(f"{col}", f"{pct:.1f}%")
                        mini_data = pd.DataFrame({'Status': ['Covered','Not Covered'], 'Count':[len(covered), total-len(covered)]})
                        fig = px.bar(mini_data, x='Status', y='Count', color='Status', color_discrete_map={'Covered':PALETTE[5],'Not Covered':PALETTE[1]})
                        st.plotly_chart(fig, use_container_width=True, key=f"deepdive_{name}_{col}")

    elif page == "State Comparison":
        st.header("State Comparison")
        all_states = sorted(set().union(*(df['State'].dropna().unique() for df in dataframes.values() if 'State' in df.columns)))
        selected_states = st.multiselect("Select States:", all_states, default=all_states[:3])
        if selected_states:
            comp = []
            for s in selected_states:
                row = {'State': s}
                for col in target_columns:
                    row[col] = any(_present_value(v) for df in dataframes.values() if 'State' in df.columns for v in df[df['State']==s].get(col, []))
                comp.append(row)
            comp_df = pd.DataFrame(comp)
            fig = go.Figure()
            for idx,row in comp_df.iterrows():
                vals = [1 if row[c] else 0 for c in target_columns]
                fig.add_trace(go.Scatterpolar(r=vals,theta=target_columns,fill='toself',name=row['State'],opacity=0.6))
            st.plotly_chart(fig, use_container_width=True, key="state_comp_radar")
            comp_df_display = comp_df.copy()
            for c in target_columns: comp_df_display[c] = comp_df_display[c].map({True:"✓",False:"✗"})
            st.dataframe(comp_df_display, use_container_width=True, hide_index=True, key="state_comp_table")

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;'>Disaster Law Dashboard<br>Focus Areas: Equity Initiatives • Mutual Aid • Mitigation Planning • Local Emergency Powers • Vulnerable Populations</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
