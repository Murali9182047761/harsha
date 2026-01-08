import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import seaborn as sns # Keeping for palette reference if needed, but not primarily used now
import matplotlib.pyplot as plt
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
            fig = px.histogram(filtered_df, x='Total_Screen_Time', nbins=20, title='Distribution of Total Screen Time', color_discrete_sequence=['#00ADB5'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        with row1_col2:
            st.markdown("### Mental Wellness Score vs Screen Time")
            fig = px.scatter(
                filtered_df, 
                x='Total_Screen_Time', 
                y='Mental_Wellness_Score', 
                color='Risk_Category',
                title='Wellness Score vs. Screen Time',
                color_discrete_map={'Low Risk': 'green', 'Moderate Risk': 'orange', 'High Risk': 'red', 'Other': 'blue'}
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            st.markdown("### Risk Category Distribution")
            risk_counts = filtered_df['Risk_Category'].value_counts().reset_index()
            risk_counts.columns = ['Risk_Category', 'Count']
            fig = px.bar(risk_counts, x='Risk_Category', y='Count', title='Count by Risk Category', color='Risk_Category')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
            
        with row2_col2:
             st.markdown("### Screen Time by Gender")
             fig = px.box(filtered_df, x='Gender', y='Total_Screen_Time', title='Total Screen Time by Gender', color='Gender')
             fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
             st.plotly_chart(fig, use_container_width=True)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title='Correlation Heatmap')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.info("Not enough numeric data for correlation heatmap.")

    # --- Prediction Section ---
    st.divider()
    st.header("🔮 Predict Your Wellness")
    st.markdown("Enter your daily habits to predict your Mental Wellness Score and Risk Category.")

    if df is not None:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split

        @st.cache_resource
        def train_models(data):
            train_df = data.copy()
            le_gender = LabelEncoder()
            train_df['Gender_Encoded'] = le_gender.fit_transform(train_df['Gender'])
            
            features = ['Age', 'Gender_Encoded', 'Work_Screen_Time', 'Social_Media_Hours', 
                        'Gaming_Hours', 'Total_Screen_Time', 'Sleep_Hours', 'Physical_Activity_Hours']
            
            X = train_df[features]
            y_score = train_df['Mental_Wellness_Score']
            y_risk = train_df['Risk_Category']
            
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_reg.fit(X, y_score)
            
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_clf.fit(X, y_risk)
            
            return rf_reg, rf_clf, le_gender, train_df

        rf_regressor, rf_classifier, gender_encoder, processed_df = train_models(df)

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=10, max_value=100, value=25)
                gender = st.pills("Gender", options=gender_encoder.classes_, selection_mode="single", default=gender_encoder.classes_[0])
                sleep_hours = st.slider("Sleep Hours", 0.0, 24.0, 7.0, 0.5)
                phys_act = st.slider("Physical Activity (Hours)", 0.0, 24.0, 1.0, 0.1)
                
            with col2:
                work_time = st.slider("Work Screen Time (Hours)", 0.0, 24.0, 4.0, 0.5)
                social_time = st.slider("Social Media (Hours)", 0.0, 24.0, 2.0, 0.5)
                gaming_time = st.slider("Gaming (Hours)", 0.0, 24.0, 1.0, 0.5)
                
            submitted = st.form_submit_button("Predict Wellness", type="primary")
            
            if submitted:
                # Basic Calculation
                total_screen_time = work_time + social_time + gaming_time
                if not gender: gender = gender_encoder.classes_[0] # Fallback
                gender_en = gender_encoder.transform([gender])[0]
                
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Gender_Encoded': [gender_en],
                    'Work_Screen_Time': [work_time],
                    'Social_Media_Hours': [social_time],
                    'Gaming_Hours': [gaming_time],
                    'Total_Screen_Time': [total_screen_time],
                    'Sleep_Hours': [sleep_hours],
                    'Physical_Activity_Hours': [phys_act]
                })
                
                predicted_score = rf_regressor.predict(input_data)[0]
                predicted_risk = rf_classifier.predict(input_data)[0]
                
                st.success("Analysis Complete!")

                # --- Results Visualization ---
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    # Gauge Chart for Score
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = predicted_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Wellness Score", 'font': {'size': 24, 'color': 'white'}},
                        number = {'font': {'size': 40, 'color': 'white'}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'color': 'white', 'size': 14}},
                            'bar': {'color': "#00ADB5"},
                            'steps': [
                                {'range': [0, 50], 'color': "#FF4B4B"},
                                {'range': [50, 75], 'color': "orange"},
                                {'range': [75, 100], 'color': "#00CC96"}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    st.markdown(f"<h3 style='text-align: center; color: white;'>Risk Category: {predicted_risk}</h3>", unsafe_allow_html=True)
                
                with res_col2:
                    # Comparison Charts (User vs Average)
                    st.markdown("#### Your Habits vs. Population Average")
                    
                    # Calculate Averages (Overall)
                    avg_data = processed_df[['Work_Screen_Time', 'Social_Media_Hours', 'Gaming_Hours', 'Sleep_Hours', 'Physical_Activity_Hours']].mean()
                    
                    categories = ['Work Screen', 'Social Media', 'Gaming', 'Sleep', 'Physical Act.']
                    user_values = [work_time, social_time, gaming_time, sleep_hours, phys_act]
                    avg_values = [avg_data['Work_Screen_Time'], avg_data['Social_Media_Hours'], avg_data['Gaming_Hours'], avg_data['Sleep_Hours'], avg_data['Physical_Activity_Hours']]
                    
                    # Radar Chart
                    fig_radar = go.Figure()

                    fig_radar.add_trace(go.Scatterpolar(
                        r=user_values,
                        theta=categories,
                        fill='toself',
                        name='You',
                        line_color='#00ADB5'
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=avg_values,
                        theta=categories,
                        fill='toself',
                        name='Average',
                        line_color='gray',
                        opacity=0.5
                    ))

                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max(max(user_values), max(avg_values)) + 2],
                                tickfont=dict(color='white', size=12),
                                gridcolor='#393E46' # subtle grid
                            ),
                            angularaxis=dict(
                                tickfont=dict(color='white', size=14),
                                gridcolor='#393E46'
                            ),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        showlegend=True,
                        legend=dict(font=dict(color="white", size=12)),
                        plot_bgcolor='rgba(0,0,0,0)', 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        font_color='white',
                        height=400
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                if predicted_score < 60:
                     st.warning("⚠️ Insight: Your predicted wellness is lower than optimal. Check your balance of screen time vs. sleep.")
                elif predicted_score > 80:
                     st.balloons()
                     st.success("🌟 Insight: You're doing great! Your habits align well with high mental wellness.")
                else:
                     st.info("ℹ️ Insight: Your habits are moderate. Small adjustments to physical activity could help.")

if __name__ == '__main__':
    main()
