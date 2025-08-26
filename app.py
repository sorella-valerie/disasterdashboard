"""
Emergency Management Dashboard
A Streamlit dashboard for analyzing emergency management data across US regions
focusing on Equity Initiatives, Mutual Aid, Mitigation Planning, 
Local Emergency Powers, and Vulnerable Populations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# Page configuration (no icons/emojis)
st.set_page_config(
    page_title="Disaster Law Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (square card layout for charts)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4788;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .chart-card {
        background-color: #f7f8fa;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        height: 520px; /* square-ish height for uniformity */
        display: flex;
        flex-direction: column;
    }
    .chart-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.05rem;
        font-weight: 600;
        color: #222;
    }
    .chart-wrap {
        flex: 1;
        min-height: 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.05rem;
    }
    /* Align checkbox with multiselect */
    div[data-testid="column"] > div > div > div > div[data-testid="stCheckbox"] {
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Title (no emoji)
st.markdown('<h1 class="main-header">Emergency Management Dashboard</h1>', unsafe_allow_html=True)

# -----------------------
# Data loading
# -----------------------
@st.cache_data
def load_data():
    """Load Excel files and return a dict of dataframes (excluding 'Comparison' and 'Midwest-Emergency')."""
    data_files = {
        'AK-HI': 'data/AKHIKeyStatutesCodesLocalAuthorityNotableProvisionsVulnerablePopulationsProtections.xlsx',
        'Appalachia-Central': 'data/AppalachiaCentralKeyStatutesCodesLocalAuthorityNotableProvisionsVulnerablePopulationsProtections.xlsx',
        'CA-WA-OR': 'data/CAWAOREmergencyDeclarationLocalEmergencyPowersMitigationPlanningMutualAidExplicitVulnerablePopulationProvisionsEquityInitiatives.xlsx',
        'Midwest-State': 'data/MidwestStateEmergencyDeclarationLocalEmergencyPowersMitigationPlanningMutualAidExplicitVulnerablePopulationsProtections.xlsx',
        'Mountain-West': 'data/MTNWestKeyStatutesCodesLocalAuthorityNotableProvisionsVulnerablePopulationsProtections.xlsx',
        'Northeast': 'data/NortheastKeyStatutesCodesLocalAuthorityNotableProvisionsVulnerablePopulationsProtections 2.xlsx',
        'South-MidAtlantic': 'data/South MidAtlanticKeyStatutesCodesLocalAuthorityNotableProvisions.xlsx',
        'Southwest': 'data/SWEmergencyDeclarationLocalEmergencyPowersMitigationPlanningMutualAidExplicitVulnerablePopulationProvisionsEquityInitiatives 2.xlsx'
    }

    dataframes = {}
    missing_files = []

    for key, filename in data_files.items():
        try:
            if Path(filename).exists():
                df = pd.read_excel(filename)
                df.columns = df.columns.str.strip()
                if 'State/Territory' in df.columns:
                    df.rename(columns={'State/Territory': 'State'}, inplace=True)
                dataframes[key] = df
            else:
                missing_files.append(filename)
        except Exception as e:
            st.warning(f"Could not load {filename}: {str(e)}")

    if missing_files:
        st.info(f"Note: {len(missing_files)} files not found. Available files loaded successfully.")

    return dataframes

@st.cache_data
def prepare_combined_data(dataframes):
    """Combine relevant dataframes for comprehensive analysis."""
    # Keep multi-category emergency files but exclude Comparison & Midwest-Emergency
    emergency_files = ['CA-WA-OR', 'Southwest', 'Midwest-State']

    combined_emergency = []
    for key in emergency_files:
        if key in dataframes:
            df = dataframes[key].copy()
            df['Region'] = key.replace('-', ' ')
            combined_emergency.append(df)

    if combined_emergency:
        combined_df = pd.concat(combined_emergency, ignore_index=True)

        # Standardize column names for analysis
        column_mapping = {
            'Explicit Vulnerable Population Provisions': 'Vulnerable Populations',
            'Vulnerable Population Provisions': 'Vulnerable Populations',
            'Explicit Vulnerable Populations Protections': 'Vulnerable Populations',
            'Equity/Health Disparities Focus': 'Equity Initiatives',
            'State Emergency Declaration': 'Emergency Declaration'
        }
        combined_df.rename(columns=column_mapping, inplace=True)

        return combined_df
    return pd.DataFrame()

# -----------------------
# Analysis helpers
# -----------------------
def _presence_mask(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Return a 1-D boolean mask: True if the row has a 'present' value
    in ANY duplicate column named `col_name`."""
    if col_name not in df.columns:
        return pd.Series(False, index=df.index)

    # Slice can be Series (unique) or DataFrame (duplicates)
    block = df[col_name]
    if isinstance(block, pd.Series):
        block = block.to_frame()

    def _present(v):
        if pd.isna(v):
            return False
        s = str(v).strip().lower()
        return s not in ("", "none", "nan")

    present_df = block.applymap(_present)
    mask = present_df.any(axis=1)  # row-wise "any present"
    # Ensure flat boolean Series aligned to df.index
    return mask.astype(bool).reindex(df.index, fill_value=False)

def analyze_coverage(df, columns_of_interest):
    """Analyze which states have coverage for each category; robust to duplicate columns."""
    coverage = {}
    if df is None or df.empty:
        return coverage

    has_state = 'State' in df.columns
    total_states = int(df['State'].nunique()) if has_state else int(len(df))

    for col in columns_of_interest:
        if col not in df.columns:
            continue

        mask = _presence_mask(df, col)

        if has_state:
            states_with_cov = int(df.loc[mask, 'State'].dropna().nunique())
        else:
            states_with_cov = int(mask.sum())

        cov_pct = float((states_with_cov / total_states * 100) if total_states > 0 else 0.0)

        coverage[col] = {
            'Total States': total_states,
            'States with Coverage': states_with_cov,
            'Coverage Percentage': cov_pct
        }
    return coverage

def create_heatmap_data(df, columns_of_interest):
    """Create heatmap data (handles duplicate columns by coalescing presence)."""
    if df is None or df.empty or 'State' not in df.columns:
        return pd.DataFrame(columns=['State', 'Category', 'Coverage'])

    pieces = []
    for col in columns_of_interest:
        if col not in df.columns:
            continue
        mask = _presence_mask(df, col)  # True if present in ANY duplicate column
        tmp = pd.DataFrame({'State': df['State'], 'Coverage': mask.astype(int)})
        # reduce to per-state coverage (if multiple rows per state, take max)
        tmp = tmp.groupby('State', as_index=False)['Coverage'].max()
        tmp['Category'] = col
        pieces.append(tmp[['State', 'Category', 'Coverage']])

    if pieces:
        return pd.concat(pieces, ignore_index=True)
    return pd.DataFrame(columns=['State', 'Category', 'Coverage'])

# -----------------------
# Main app
# -----------------------

# Dark color palette options - using Deep Ocean theme
PALETTE = [
    "#0d1b2a",  # very dark navy
    "#1b263b",  # dark blue
    "#415a77",  # steel blue
    "#778da9",  # light steel
    "#e0e1dd",  # light gray
    "#2a9d8f",  # teal accent
    "#457b9d",  # medium blue
]

# Alternative dark palettes (you can switch by changing PALETTE above):
# Forest Night: ["#1b2f1b", "#2d4a2b", "#3e5c3a", "#52734d", "#91c788", "#668b5a", "#4a6741"]
# Midnight Purple: ["#10002b", "#240046", "#3c096c", "#5a189a", "#7b2cbf", "#9d4edd", "#c77dff"]
# Charcoal Ember: ["#0d0d0d", "#2b2b2b", "#404040", "#595959", "#737373", "#8c8c8c", "#ff6b35"]

def main():
    # Load data
    with st.spinner('Loading data...'):
        dataframes = load_data()
        combined_df = prepare_combined_data(dataframes)

    # Sidebar navigation (Regional Analysis and Data Explorer removed)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View:",
        ["Summary", "Category Deep Dive", "State Comparison"]
    )

    # Columns of interest
    target_columns = [
        'Equity Initiatives',
        'Mutual Aid',
        'Mitigation Planning',
        'Local Emergency Powers',
        'Vulnerable Populations'
    ]

    # -----------------------
    # Summary (with dark color palette)
    # -----------------------
    if page == "Summary":
        st.subheader("Summary")

        # --- Prepare data ---
        coverage_data = analyze_coverage(combined_df, target_columns) if not combined_df.empty else {}
        coverage_df = pd.DataFrame([
            {'Category': cat,
            'Coverage %': v['Coverage Percentage'],
            'States Covered': v['States with Coverage'],
            'Total States': v['Total States']}
            for cat, v in coverage_data.items()
        ])

        heatmap_df = create_heatmap_data(combined_df, target_columns)

        region_coverage = []
        if not combined_df.empty and {'Region', 'State'}.issubset(combined_df.columns):
            for region, g in combined_df.groupby('Region', dropna=False):
                region_tot_states = g['State'].dropna().nunique()
                for cat in target_columns:
                    if cat in g.columns:
                        mask = _presence_mask(g, cat)
                        present = g.loc[mask, 'State'].dropna().nunique()
                        share = present / region_tot_states if region_tot_states else 0.0
                        region_coverage.append({'Region': region, 'Category': cat, 'Share': share})
        region_cov_df = pd.DataFrame(region_coverage)

        top_states_df = pd.DataFrame()
        if not combined_df.empty and 'State' in combined_df.columns:
            parts = []
            for cat in target_columns:
                if cat in combined_df.columns:
                    mask = _presence_mask(combined_df, cat)
                    tmp = pd.DataFrame({'State': combined_df['State'], cat: mask.astype(int)})
                    tmp = tmp.groupby('State', as_index=False)[cat].max()
                    parts.append(tmp.set_index('State'))
            if parts:
                joined = pd.concat(parts, axis=1).fillna(0).astype(int)
                joined['Total Categories'] = joined.sum(axis=1)
                top_states_df = joined['Total Categories'].sort_values(ascending=False).head(15).reset_index()
                top_states_df.columns = ['State', 'Total Categories']

        # --- Layout: 2x2 grid of cards ---
        c1, c2 = st.columns(2)

        # 1) Coverage by Category
        with c1:
            with st.container(border=True):
                st.subheader("Coverage by Category")
                if not coverage_df.empty:
                    fig = px.bar(
                        coverage_df,
                        x="Category", y="Coverage %",
                        text="Coverage %",
                        color="Category",
                        color_discrete_sequence=PALETTE
                    )
                    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
                    fig.update_layout(
                        margin=dict(l=8, r=8, t=8, b=8), 
                        height=360, 
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("No coverage data available.")

        # 2) Heatmap
        with c2:
            with st.container(border=True):
                st.subheader("State Coverage Heatmap")
                if not heatmap_df.empty:
                    pivot_df = heatmap_df.pivot(
                        index="Category", columns="State", values="Coverage"
                    ).fillna(0).astype(int)

                    fig_heatmap = px.imshow(
                        pivot_df,
                        labels=dict(x="State", y="Category", color="Coverage"),
                        color_continuous_scale=[
                            "#0d1b2a",  # very dark navy
                            "#1b263b",  # dark blue
                            "#2a9d8f",  # teal
                            "#52b788"   # light green
                        ],
                    )

                    fig_heatmap.update_layout(
                        width=800,
                        height=360,
                        xaxis=dict(side="bottom"),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )

                    st.plotly_chart(fig_heatmap, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("No heatmap data available.")

        c3, c4 = st.columns(2)

        # 3) Average Coverage by Region
        with c3:
            with st.container(border=True):
                st.subheader("Average Coverage by Region")
                if not region_cov_df.empty:
                    avg_region = region_cov_df.groupby("Region")["Share"].mean().reset_index()
                    fig_region = px.bar(
                        avg_region, x="Region", y="Share",
                        text="Share", color="Region",
                        color_discrete_sequence=PALETTE
                    )
                    fig_region.update_traces(texttemplate="%{text:.0%}", textposition="outside", cliponaxis=False)
                    fig_region.update_layout(
                        margin=dict(l=8, r=8, t=8, b=8), 
                        height=360, 
                        yaxis_tickformat=".0%",
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_region, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("No region coverage data available.")

        # 4) Top States
        with c4:
            with st.container(border=True):
                st.subheader("Top States by Categories Covered")
                if not top_states_df.empty:
                    fig_top = px.bar(
                        top_states_df.sort_values("Total Categories"),
                        x="Total Categories", y="State",
                        orientation="h", text="Total Categories",
                        color="Total Categories",
                        color_continuous_scale=[PALETTE[0], PALETTE[2], PALETTE[5]]
                    )
                    fig_top.update_traces(textposition="outside")
                    fig_top.update_layout(
                        margin=dict(l=8, r=8, t=8, b=8), 
                        height=360, 
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("No state coverage ranking available.")

    # -----------------------
    # Category Deep Dive (with aligned select all)
    # -----------------------
    elif page == "Category Deep Dive":
        st.header("Category Deep Dive")

        # Aligned select-all UI using columns
        col1, col2 = st.columns([5, 1])
        
        with col1:
            selected_categories = st.multiselect(
                "Select Categories:", 
                target_columns, 
                default=[]
            )
        
        with col2:
            # Add spacing to align checkbox with multiselect
            st.write("")  # Empty space for alignment
            select_all = st.checkbox("Select all")
        
        # If select all is checked, override the selection
        if select_all:
            selected_categories = target_columns

        if not selected_categories:
            st.info("Select at least one category (or use 'Select all').")
            return

        # Find all dataframes containing any of the selected categories
        dfs_with_category = []
        for name, df in dataframes.items():
            cols_str = [str(c) for c in df.columns]
            if any(cat in c for cat in selected_categories for c in cols_str):
                dfs_with_category.append((name, df))

        if dfs_with_category:
            st.write(f"Found in {len(dfs_with_category)} datasets:")
            tabs = st.tabs([name for name, _ in dfs_with_category])

            for tab, (name, df) in zip(tabs, dfs_with_category):
                with tab:
                    present_cols = []
                    for cat in selected_categories:
                        for col in df.columns:
                            if cat in str(col):
                                present_cols.append((cat, col))
                    if not present_cols:
                        st.warning("No matching columns found.")
                        continue

                    if 'State' in df.columns:
                        for cat, actual_col in present_cols:
                            st.subheader(f"{name} — {cat}")
                            non_empty = df[df[actual_col].notna() & (df[actual_col] != '') & (df[actual_col] != 'None')]
                            
                            # Create a simple bar chart for this category
                            coverage_pct = (len(non_empty) / len(df)) * 100 if len(df) > 0 else 0
                            
                            col_a, col_b = st.columns([1, 2])
                            with col_a:
                                st.metric("Coverage", f"{coverage_pct:.1f}%")
                                st.write(f"States with {cat}: {len(non_empty)}/{len(df)}")
                            
                            with col_b:
                                if not non_empty.empty:
                                    # Mini bar chart showing covered vs not covered
                                    mini_data = pd.DataFrame({
                                        'Status': ['Covered', 'Not Covered'],
                                        'Count': [len(non_empty), len(df) - len(non_empty)]
                                    })
                                    fig_mini = px.bar(
                                        mini_data, x='Status', y='Count',
                                        color='Status',
                                        color_discrete_map={'Covered': PALETTE[5], 'Not Covered': PALETTE[1]}
                                    )
                                    fig_mini.update_layout(
                                        height=200,
                                        showlegend=False,
                                        margin=dict(l=0, r=0, t=0, b=0)
                                    )
                                    st.plotly_chart(fig_mini, use_container_width=True, config={"displayModeBar": False})
                            
                            if not non_empty.empty:
                                with st.expander(f"View data for {len(non_empty)} states"):
                                    st.dataframe(non_empty[['State', actual_col]], use_container_width=True)
                    else:
                        st.info("This dataset has no 'State' column; showing raw column(s).")
                        disp_cols = [c for _, c in present_cols]
                        st.dataframe(df[disp_cols].head(), use_container_width=True)
        else:
            st.warning("No datasets contain the selected category(ies).")

    # -----------------------
    # State Comparison
    # -----------------------
    elif page == "State Comparison":
        st.header("State Comparison")

        # Collect all states
        all_states = set()
        for df in dataframes.values():
            if 'State' in df.columns:
                all_states.update(df['State'].dropna().unique().tolist())
        all_states = sorted(list(all_states))

        selected_states = st.multiselect(
            "Select states to compare:",
            all_states,
            default=all_states[:3] if len(all_states) >= 3 else all_states
        )

        if selected_states:
            comparison_data = []
            for state in selected_states:
                state_info = {'State': state}
                for category in target_columns:
                    found = False
                    for name, df in dataframes.items():
                        if 'State' in df.columns and state in df['State'].values:
                            # consider all rows for state
                            rows = df[df['State'] == state]
                            for col in df.columns:
                                if (category in str(col)) or (
                                    category == 'Vulnerable Populations' and 'Vulnerable' in str(col)
                                ) or (
                                    category == 'Equity Initiatives' and 'Equity' in str(col)
                                ):
                                    # if any row is non-empty for the state
                                    if rows[col].apply(lambda v: pd.notna(v) and v != '' and v != 'None').any():
                                        state_info[category] = 'Yes'
                                        found = True
                                        break
                            if found:
                                break
                    if not found:
                        state_info[category] = 'No'
                comparison_data.append(state_info)

            comparison_df = pd.DataFrame(comparison_data)

            # Create two columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("State Coverage Comparison")
                
                if len(comparison_df) > 1:
                    categories = [c for c in comparison_df.columns if c != 'State']
                    fig = go.Figure()
                    
                    # Use dark colors for radar chart
                    colors = PALETTE[:len(comparison_df)]
                    
                    for idx, row in enumerate(comparison_df.iterrows()):
                        values = [1 if row[1][cat] == 'Yes' else 0 for cat in categories]
                        fig.add_trace(go.Scatterpolar(
                            r=values, 
                            theta=categories, 
                            fill='toself', 
                            name=row[1]['State'],
                            line_color=colors[idx % len(colors)],
                            fillcolor=colors[idx % len(colors)],
                            opacity=0.6
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True, 
                                range=[0, 1],
                                gridcolor='lightgray'
                            ),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        showlegend=True,
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Coverage Matrix")
                
                # Create a styled dataframe
                display_df = comparison_df.copy()
                for col in target_columns:
                    display_df[col] = display_df[col].apply(lambda x: '✓' if x == 'Yes' else '✗')
                
                # Display with custom styling
                st.dataframe(
                    display_df, 
                    use_container_width=True, 
                    hide_index=True,
                    height=400
                )
                
                # Summary metrics
                st.write("**Summary Statistics:**")
                for state in selected_states:
                    state_row = comparison_df[comparison_df['State'] == state]
                    coverage_count = sum(1 for col in target_columns if state_row[col].values[0] == 'Yes')
                    coverage_pct = (coverage_count / len(target_columns)) * 100
                    st.write(f"- {state}: {coverage_count}/{len(target_columns)} categories ({coverage_pct:.0f}%)")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        Emergency Management Dashboard | Data Analysis Tool<br>
        Focus Areas: Equity Initiatives • Mutual Aid • Mitigation Planning • Local Emergency Powers • Vulnerable Populations
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":

    main()
