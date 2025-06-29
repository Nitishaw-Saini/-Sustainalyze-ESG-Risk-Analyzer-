# Add this at the top of your Streamlit app
import streamlit as st
import pandas as pd
import plotly.express as px

# Load your final ESG dataset
df = pd.read_csv("final_esg_dataset_with_structured_insights.csv")

# Add page selector (if not using multipage mode)
page = st.sidebar.selectbox("Choose Page", ["Single Company View", "All Companies Overview"])

# ------------------------- All Companies Dashboard -------------------------
if page == "All Companies Overview":
    st.title("🌍 ESG Overview: All Companies Dashboard")

    # 1. Overall ESG Risk Distribution
    st.subheader("📊 ESG Risk Level Distribution")
    fig1 = px.histogram(df, x="ESG_Risk_Label", color="ESG_Risk_Label", 
                        color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"})
    st.plotly_chart(fig1, use_container_width=True)

    # 2. ESG Score Comparison per Company (bar chart)
    st.subheader("🏢 Company-wise Total ESG Scores")
    fig2 = px.bar(df.sort_values("Total_ESG_Risk_Score", ascending=False),
                  x="Company Name", y="Total_ESG_Risk_Score", 
                  color="ESG_Risk_Label",
                  color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"},
                  title="Company-wise ESG Risk Scores")
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Sector-wise ESG Risk Averages
    st.subheader("🏭 Average ESG Score by Sector")
    sector_df = df.groupby("Sector")["Total_ESG_Risk_Score"].mean().reset_index()
    fig3 = px.bar(sector_df.sort_values("Total_ESG_Risk_Score", ascending=True),
                  x="Sector", y="Total_ESG_Risk_Score", color="Total_ESG_Risk_Score",
                  color_continuous_scale="RdYlGn_r")
    fig3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

    # 4. Controversy Score Distribution
    st.subheader("🚨 Controversy Score Distribution")
    fig4 = px.box(df, x="Sector", y="Controversy_Score", points="all",
                  color="Sector", title="Controversy Scores by Sector")
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)

    st.info("Use filters or switch to 'Single Company View' from the sidebar for detailed ESG metrics.")

# ------------------------- Single Company Dashboard (your existing code) -------------------------
elif page == "Single Company View":
      # Sidebar – Company Selection
st.sidebar.title("📊 ESG Risk Analyzer")

# 🎯 Sidebar Filters
st.sidebar.header("🔎 Filter Companies")

# Filter widgets: Sector and Risk Level only
selected_sector = st.sidebar.selectbox("Select Sector", ['All'] + sorted(df['Sector'].dropna().unique().tolist()))
selected_risk = st.sidebar.selectbox("Select ESG Risk Level", ['All'] + sorted(df['ESG_Risk_Level'].dropna().unique().tolist()))

# Apply filters to create a filtered DataFrame
filtered_df = df.copy()
if selected_sector != 'All':
    filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]
if selected_risk != 'All':
    filtered_df = filtered_df[filtered_df['ESG_Risk_Level'] == selected_risk]

# Display number of matching results
st.sidebar.markdown(f"🔢 **Filtered Results:** {len(filtered_df)} Companies")

# ✅ Check for empty filter result
if filtered_df.empty:
    st.warning(f"⚠️ No companies available for Sector **{selected_sector}** and ESG Risk Level **{selected_risk}**.")
    st.stop()  # 🛑 Stops further rendering
else:
    # Dynamic company dropdown based on filters
    company = st.sidebar.selectbox("Choose a Company", sorted(filtered_df['Company Name'].unique()))

    # Pull selected company data
    info = filtered_df[filtered_df["Company Name"] == company].iloc[0]

# Header
st.title("🌍 ESG Risk Dashboard")
st.markdown(f"### Company: **{company}**")


# Extract input features from the selected company's row
input_features = [
    info['Total_ESG_Risk_Score'],
    info['Predicted_ESG_Score'],
    info['ESG_Risk_Exposure'],
    info['ESG_Risk_Management'],
    info['Controversy_Score'],
    info['Sector_encoded'],
    info['Industry_encoded'],
    info['Controversy_Level_encoded']
]

predicted_label = model.predict([input_features])[0]

# Map label to risk level and emoji
risk_level_map = {
    0: ("Low", "🟢"),
    1: ("Medium", "🟠"),
    2: ("High", "🔴")
}
risk_label, color_emoji = risk_level_map.get(predicted_label)

# ✅ Update info dict so KPI uses this
info['ESG_Risk_Label'] = risk_label


# --- 1. KPI Cards
col1, col2, col3,col4 = st.columns(4)
col1.metric("Total ESG Risk Score", info['Total_ESG_Risk_Score'])
col2.metric("Predicted ESG Score", round(info['Predicted_ESG_Score'], 2))
col3.metric("Actual Risk Level", info['ESG_Risk_Level'])
col4.metric("Predicted Risk Level (ML)", risk_label)


# --- 2. Color-coded Risk Exposure Gauge ---
st.subheader("🎯 ESG Risk Exposure")

exposure_score = info['ESG_Risk_Exposure']
rounded_exposure = round(exposure_score, 2)

import plotly.graph_objects as go
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=rounded_exposure,
    title={'text': "ESG Risk Exposure"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "black"},
        'steps': [
            {'range': [0, 30], 'color': "green"},
            {'range': [30, 50], 'color': "orange"},
            {'range': [50, 70], 'color': "red"},
            {'range': [70, 100], 'color': "darkred"},
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': rounded_exposure
        }
    }
))

st.plotly_chart(fig_gauge, use_container_width=True)



# --- 5. Expandable ESG Definitions
with st.expander("💡 What is ESG Risk Score?"):
    st.markdown("ESG Risk Score measures how exposed a company is to environmental, social, and governance risks and how well it manages them." 
" A lower score indicates better ESG practices and lower risk; a higher score suggests poor management and greater exposure."
" It's used by investors, regulators, and analysts to evaluate a company's long-term sustainability and ethical impact.")

# --- 3. ESG Component Bar Chart
st.subheader("📊 ESG Score Breakdown")

import plotly.graph_objects as go

# ESG component scores
env_score = round(info['Environment_Score'], 1)
soc_score = round(info['Social_Score'], 1)
gov_score = round(info['Governance_Score'], 1)

# Create horizontal stacked bar chart with value labels
fig_esg = go.Figure()

fig_esg.add_trace(go.Bar(
    y=["ESG Component Averages"],
    x=[env_score],
    name='Environment',
    orientation='h',
    marker=dict(color="#2E8B57"),
    text=[env_score],
    textposition="inside",
    hovertemplate='Environment: %{x}<extra></extra>'
))

fig_esg.add_trace(go.Bar(
    y=["ESG Component Averages"],
    x=[soc_score],
    name='Social',
    orientation='h',
    base=[env_score],
    marker=dict(color="#008080"),
    text=[soc_score],
    textposition="inside",
    hovertemplate='Social: %{x}<extra></extra>'
))

fig_esg.add_trace(go.Bar(
    y=["ESG Component Averages"],
    x=[gov_score],
    name='Governance',
    orientation='h',
    base=[env_score + soc_score],
    marker=dict(color="#DAA520"),
    text=[gov_score],
    textposition="inside",
    hovertemplate='Governance: %{x}<extra></extra>'
))

fig_esg.update_layout(
    barmode='stack',
    title="ESG Component Averages",
    xaxis_title="Score",
    yaxis_title="",
    height=250,
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')  # Optional: for dark themes
)

st.plotly_chart(fig_esg, use_container_width=True)



# --- 4. Sector & Industry Comparison (avg)
st.subheader("🏭 Sector & Industry Information")

sector = info['Sector']
industry = info['Industry']

sector_avg = df[df['Sector'] == sector]['Total_ESG_Risk_Score'].mean()
industry_avg = df[df['Industry'] == industry]['Total_ESG_Risk_Score'].mean()

st.write(
    f"**Sector ({sector}) Avg Risk Score:** {sector_avg:.2f}  \n"
    f"**Industry ({industry}) Avg Risk Score:** {industry_avg:.2f}"
)

# --- 6. ESG Score Comparison – Company vs Sector vs Industry
st.subheader("📈 ESG Score Comparison: Company vs Sector vs Industry")

comparison_df = pd.DataFrame({
    "Category": ["Selected Company", f"{info['Sector']} Avg", f"{info['Industry']} Avg"],
    "ESG Score": [
        info['Total_ESG_Risk_Score'],
        df[df['Sector'] == info['Sector']]['Total_ESG_Risk_Score'].mean(),
        df[df['Industry'] == info['Industry']]['Total_ESG_Risk_Score'].mean()
    ]
})

import plotly.express as px
fig = px.bar(
    comparison_df,
    x="Category",
    y="ESG Score",
    color="Category",
    title=f"ESG Score Comparison for {company}",
    text_auto=True,
    color_discrete_map={
        "Selected Company": "#556B2F", 
        f"{info['Sector']} Avg": "#468499",  f"{info['Industry']} Avg": "#B8860B"     }
)

st.plotly_chart(fig, use_container_width=True)

# --- 4.1 AI-Style ESG Comparison Insight
relative_sector_diff = sector_avg - info['Total_ESG_Risk_Score']
relative_industry_diff = industry_avg - info['Total_ESG_Risk_Score']

insight_text = (
    f"{company} demonstrates stronger ESG performance compared to its sector "
    f"({info['Sector']}) and industry ({info['Industry']}). The ESG risk score is "
    f"{relative_sector_diff:.1f} points lower than the sector average and "
    f"{relative_industry_diff:.1f} points lower than the industry average."
)

st.info(f"💡 **Insight:** {insight_text}")

import plotly.graph_objects as go

# --- Controversy Score Section ---
st.subheader("🚨 Controversy Score Analysis")

# Get score
controversy_score = info['Controversy_Score']
rounded_controversy = round(controversy_score, 2)

# Determine color and interpretation
if controversy_score <= 20:
    color = "green"
    interpretation = "🟢 Minimal controversy — good public and regulatory standing."
elif controversy_score <= 40:
    color = "yellow"
    interpretation = "🟡 Some reported incidents or mild concerns."
elif controversy_score <= 60:
    color = "orange"
    interpretation = "🟠 Notable issues in media or governance."
else:
    color = "red"
    interpretation = "🔴 Repeated or severe controversies — reputational risk."

# Bar Chart
fig_bar = go.Figure(go.Bar(
    x=[rounded_controversy],
    y=["Controversy Score"],
    orientation='h',
    marker_color=color,
    text=[f"{rounded_controversy}/100"],
    textposition='auto'
))

fig_bar.update_layout(
    xaxis=dict(range=[0, 100]),
    height=150,
    margin=dict(l=20, r=20, t=40, b=20),
    showlegend=False
)

# Display Chart + Interpretation
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown(f"### ℹ️ Interpretation\n{interpretation}")


# --- 10. Insight Summary
st.subheader("🧠 ESG Insight Summary")
st.markdown(f"Based on its current ESG risk score of **{info['Total_ESG_Risk_Score']}**, "
            f"{company} is demonstrating **{info['ESG_Risk_Management']}** management practices "
            f"with an exposure level of **{info['ESG_Risk_Exposure']}**. The predicted future score of "
            f"**{info['Predicted_ESG_Score']}** suggests that ESG performance is expected to "
            f"{'improve' if info['Predicted_ESG_Score'] > info['Total_ESG_Risk_Score'] else 'decline'}. "
            f"Controversy level is marked as **{info['Controversy_Level']}**, indicating associated risks.")

# --- ML Prediction (Real-Time)
# Load your trained Random Forest model
import joblib
model = joblib.load("rf_esg_model.pkl")

# Define the selected company from the filtered DataFrame
info = df[df["Company Name"] == company].iloc[0]

# Extract input features from the selected company's row
input_features = [
    info['Total_ESG_Risk_Score'],
    info['Predicted_ESG_Score'],
    info['ESG_Risk_Exposure'],
    info['ESG_Risk_Management'],
    info['Controversy_Score'],
    info['Sector_encoded'],
    info['Industry_encoded'],
    info['Controversy_Level_encoded']
]

# Predict using the ML model

# Show result
st.subheader("🔍 Real-Time ESG Risk Prediction")
st.markdown(
    f"📌 **Predicted ESG Risk Level (ML): {predicted_label}** {color_emoji} "
    f"which is classified as **{risk_label}**"
)
st.caption("🟢 Low | 🟠 Medium | 🔴 High ESG Risk Levels")


# --- 10. Actual vs Predicted ESG Score (Bar Chart)
st.subheader("📊 Actual vs Predicted ESG Score")

# Prepare data
actual_vs_pred_df = pd.DataFrame({
    'Score Type': ['Actual ESG Risk Score', 'Predicted ESG Score'],
    'Score Value': [info['Total_ESG_Risk_Score'], info['Predicted_ESG_Score']]
})

# Bar Chart

fig_actual_vs_pred = px.bar(
    x=["Actual ESG Score", "Predicted ESG Score"],
    y=[info['Total_ESG_Risk_Score'], info['Predicted_ESG_Score']],
    color=["Actual", "Predicted"],
    title=f"Actual vs Predicted ESG Score for {company}",
    color_discrete_map={
        "Actual": "#3C6E47",      # Evergreen (Actual)
        "Predicted": "#7FB3A6"    # Aqua Gray (Predicted)
    },
    text_auto=True
)

st.plotly_chart(fig_actual_vs_pred, use_container_width=True)


# --- 1.5 ESG Score Summary Tile
actual_score = round(info['Total_ESG_Risk_Score'], 2)
predicted_score = round(info['Predicted_ESG_Score'], 2)
difference = predicted_score - actual_score

# Define summary message
if difference > 0:
    summary_text = f"🔺 ESG performance may **worsen** by {abs(difference):.2f} points."
    summary_color = "red"
elif difference < 0:
    summary_text = f"🟢 ESG performance is expected to **improve** by {abs(difference):.2f} points."
    summary_color = "green"
else:
    summary_text = "🟡 ESG performance is expected to **remain stable**."
    summary_color = "orange"

# Show the tile
st.markdown(f"""
<div style='padding: 0.5rem; border-radius: 0.5rem; background-color: {summary_color}; color: white; text-align: center; font-size: 1rem;'>
    <strong>Actual vs Predicted ESG Score:</strong><br>{summary_text}
</div>
""", unsafe_allow_html=True)



# --- 9. Download ESG Report
csv_download = df[df['Company Name'] == company].to_csv(index=False).encode('utf-8')
st.download_button("📥 Download ESG Report", data=csv_download, file_name=f"{company}_esg_report.csv")

 pass
