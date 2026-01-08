import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(
    page_title="Screen Time Wellness Analysis",
    page_icon="🧠",
    layout="wide"
)

# Function to load data
@st.cache_data
def load_data():
    # Construct absolute path to dataset
    # Assuming app/dashboard.py is running, data is in ../data/raw/dataset.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'raw', 'dataset.csv')
    
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found at path: {data_path}")
        return None

def main():
    # Custom CSS for better aesthetics
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #FAFAFA;
        }
        h1, h2, h3 {
            color: #00ADB5 !important;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stMetric {
            background-color: #222831;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #393E46;
        }
        .stMetricLabel {
            color: #EEEEEE !important;
        }
        .stMetricValue {
            color: #00FFF5 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🧠 Screen Time & Mental Wellness Analysis")
    st.markdown("""
    This application analyzes the relationship between screen time habits and mental wellness scores.
    Explore the data, visualize trends, and understand the impact of digital consumption on well-being.
    """)

    df = load_data()

    if df is not None:
        # Set Plot Style
        plt.style.use('dark_background')
        sns.set_palette("bright")

        # Sidebar - Filters
        st.sidebar.header("Filter Data")
        
        # Gender Filter
        selected_genders = st.sidebar.multiselect(
            "Select Gender",
            options=df['Gender'].unique(),
            default=df['Gender'].unique()
        )
        
        # Risk Category Filter
        selected_risks = st.sidebar.multiselect(
            "Select Risk Category",
            options=df['Risk_Category'].unique(),
            default=df['Risk_Category'].unique()
        )
        
        # Apply Filters
        filtered_df = df[
            (df['Gender'].isin(selected_genders)) & 
            (df['Risk_Category'].isin(selected_risks))
        ]

        # KPIS
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Wellness Score", f"{filtered_df['Mental_Wellness_Score'].mean():.2f}")
        col2.metric("Avg Total Screen Time", f"{filtered_df['Total_Screen_Time'].mean():.2f} h")
        col3.metric("Avg Sleep Hours", f"{filtered_df['Sleep_Hours'].mean():.2f} h")
        col4.metric("Total Participants", len(filtered_df))

        # Dataset Preview
        with st.expander("Values Data Preview"):
            st.dataframe(filtered_df.head())

        # Visualizations
        st.subheader("Data Visualizations")
        
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.markdown("### Total Screen Time Distribution")
            fig, ax = plt.subplots()
            sns.histplot(filtered_df['Total_Screen_Time'], kde=True, ax=ax, color='skyblue')
            ax.set_title("Distribution of Total Screen Time")
            st.pyplot(fig)

        with row1_col2:
            st.markdown("### Mental Wellness Score vs Screen Time")
            fig, ax = plt.subplots()
            sns.scatterplot(data=filtered_df, x='Total_Screen_Time', y='Mental_Wellness_Score', hue='Risk_Category', ax=ax)
            ax.set_title("Wellness Score vs. Screen Time")
            st.pyplot(fig)

        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.markdown("### Risk Category Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=filtered_df, x='Risk_Category', palette='viridis', ax=ax)
            ax.set_title("Count of Participants by Risk Category")
            st.pyplot(fig)
            
        with row2_col2:
             st.markdown("### Screen Time by Gender")
             fig, ax = plt.subplots()
             sns.boxplot(data=filtered_df, x='Gender', y='Total_Screen_Time', ax=ax)
             ax.set_title("Total Screen Time by Gender")
             st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        # Select numeric columns only
        numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
             st.info("Not enough numeric data for correlation heatmap.")

if __name__ == '__main__':
    main()
