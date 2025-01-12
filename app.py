import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.file_handler import save_file, extract_data
from utils.anomaly_detection import detect_anomalies
from utils.data_preprocessing import DataPreprocessor
from utils.chat_assistant import ChatAssistant
from utils.climate_data import ClimateDataRetriever

# Set page configuration
st.set_page_config(
    page_title="Climate Data Validator",
    page_icon="🌍",
    layout="wide"
)

# Define plotting functions
def plot_time_series(data, column, anomaly_indices=None):
    """Plot time series data with highlighted anomalies."""
    fig, ax = plt.subplots(figsize=(12, 6))
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    ax.plot(data.index, data[column], label='Normal', alpha=0.7)
    if anomaly_indices and len(anomaly_indices) > 0:
        if isinstance(anomaly_indices[0], (pd.Timestamp, str)):
            anomaly_indices = data.index.get_indexer(anomaly_indices)
        ax.scatter(data.index[anomaly_indices], data.iloc[anomaly_indices][column],
                  color='red', label='Anomaly', zorder=5)
    ax.set_title(f'{column} Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel(column)
    plt.xticks(rotation=45)
    plt.legend()
    return fig

def plot_distribution(data, column):
    """Plot distribution of values with kernel density estimation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x=column, kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    return fig

def plot_correlation_heatmap(data):
    """Plot correlation heatmap for numeric columns."""
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        plt.title('Correlation Heatmap')
        return fig
    return None

def plot_missing_values(data):
    """Plot missing values heatmap."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    return fig

# Set style for plots
plt.style.use('default')
sns.set_style("whitegrid")

# Initialize components
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = DataPreprocessor()
if 'chat_assistant' not in st.session_state:
    st.session_state.chat_assistant = ChatAssistant()
if 'climate_retriever' not in st.session_state:
    st.session_state.climate_retriever = ClimateDataRetriever()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_context' not in st.session_state:
    st.session_state.chat_context = None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Climate Data Download", "Developer Section"])

if page == "Data Upload":
    # Main title
    st.title("Climate Data Validator")
    st.write("""
    This AI-powered system helps validate and verify climate datasets by:
    - Checking data structure and completeness
    - Detecting anomalies using multiple methods
    - Validating temporal consistency
    - Analyzing correlations between variables
    """)

    # Add user manual button in sidebar
    with st.sidebar:
        st.title("Documentation")
        if st.button("📖 View User Manual"):
            try:
                with open("docs/user_manual.md", "r") as f:
                    manual_content = f.read()
                st.markdown(manual_content)
            except Exception as e:
                st.error("Could not load user manual. Please make sure the file exists.")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    detection_threshold = st.sidebar.slider(
        "Anomaly Detection Sensitivity",
        min_value=0.01,
        max_value=0.2,
        value=0.1,
        help="Lower values mean stricter anomaly detection"
    )

    # File upload widget
    uploaded_file = st.file_uploader("Choose a climate dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        # Save and process uploaded file
        filepath = save_file(uploaded_file, UPLOAD_FOLDER)
        if filepath:
            st.success(f"File uploaded successfully: {uploaded_file.name}")

            # Extract data
            data = extract_data(filepath)
            if data is not None:
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(data.head() if isinstance(data, pd.DataFrame) else pd.DataFrame(data[:5]))

                # Process data
                if isinstance(data, pd.DataFrame):
                    # Preprocess data
                    processed_data, validation_report = st.session_state.preprocessor.process_data(data)
                    
                    # Data Overview Section
                    st.subheader("Data Overview")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Dataset Shape:", processed_data.shape)
                        st.write("Numeric Columns:", len(processed_data.select_dtypes(include=['float64', 'int64']).columns))
                    
                    with col2:
                        st.write("Total Missing Values:", processed_data.isnull().sum().sum())
                        st.write("Duplicate Rows:", processed_data.duplicated().sum())

                    # Missing Values Visualization
                    st.subheader("Missing Values Analysis")
                    if processed_data.isnull().sum().sum() > 0:
                        st.pyplot(plot_missing_values(processed_data))
                        st.write("Missing values summary:")
                        st.write(processed_data.isnull().sum())
                    else:
                        st.success("✓ No missing values found")

                    # Time Series Analysis
                    if 'timestamp' in processed_data.columns:
                        st.subheader("Time Series Analysis")
                        numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
                        numeric_cols = [col for col in numeric_cols if col != 'timestamp']
                        
                        if numeric_cols:  # Only show if there are numeric columns
                            selected_col = st.selectbox("Select variable for time series analysis:", 
                                                    numeric_cols)
                            
                            if selected_col:
                                # Create a copy of the data for time series analysis
                                ts_data = processed_data.copy()
                                ts_data['timestamp'] = pd.to_datetime(ts_data['timestamp'])
                                ts_data.set_index('timestamp', inplace=True)
                                
                                # Get anomaly indices
                                anomaly_indices = []
                                if validation_report['range_anomalies'].get(selected_col):
                                    # Convert numeric indices to timestamps
                                    anomaly_indices = ts_data.index[validation_report['range_anomalies'][selected_col]].tolist()
                                
                                st.pyplot(plot_time_series(ts_data, selected_col, anomaly_indices))

                    # Distribution Analysis
                    st.subheader("Distribution Analysis")
                    dist_col = st.selectbox("Select variable for distribution analysis:",
                                          processed_data.select_dtypes(include=['float64', 'int64']).columns)
                    if dist_col:
                        st.pyplot(plot_distribution(processed_data, dist_col))

                    # Correlation Analysis
                    st.subheader("Correlation Analysis")
                    corr_fig = plot_correlation_heatmap(processed_data)
                    if corr_fig:
                        st.pyplot(corr_fig)
                    else:
                        st.info("Not enough numeric columns for correlation analysis")

                    # Data Validation Results Section
                    st.subheader("Data Validation Results")
                    
                    # Structure Issues
                    if validation_report['structure_issues']:
                        st.error("⚠️ Structure Issues Found:")
                        for issue in validation_report['structure_issues']:
                            st.write(f"- {issue}")
                    else:
                        st.success("✅ Data structure validation passed!")

                    # Range Anomalies
                    if validation_report['range_anomalies']:
                        st.warning("🔍 Range Anomalies Detected:")
                        for column, indices in validation_report['range_anomalies'].items():
                            st.write(f"- {column}: {len(indices)} values outside expected range")
                    else:
                        st.success("✅ No range anomalies detected")

                    # Temporal Inconsistencies
                    if validation_report['temporal_inconsistencies']:
                        st.warning("⚠️ Temporal Inconsistencies Found:")
                        st.write(f"- {len(validation_report['temporal_inconsistencies'])} time gaps detected")
                    else:
                        st.success("✅ No temporal inconsistencies found")

                    # Duplicates
                    if validation_report['duplicates']:
                        st.warning("🔄 Duplicate Records Found:")
                        st.write(f"- {len(validation_report['duplicates'])} duplicate entries detected")
                    else:
                        st.success("✅ No duplicates found")

                    # Run anomaly detection
                    st.subheader("Advanced Anomaly Detection")
                    anomaly_report = detect_anomalies(processed_data)
                    
                    # Statistical Anomalies
                    if anomaly_report['statistical_anomalies']:
                        st.warning("Statistical Anomalies (Z-score > 3):")
                        for col, indices in anomaly_report['statistical_anomalies'].items():
                            if indices:
                                st.write(f"- {col}: {len(indices)} anomalies")
                                # Plot statistical anomalies
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.boxplot(data=processed_data, y=col, ax=ax)
                                sns.stripplot(data=processed_data.iloc[indices], y=col,
                                            color='red', size=10, ax=ax)
                                plt.title(f'Statistical Anomalies in {col}')
                                st.pyplot(fig)
                                st.dataframe(processed_data.loc[indices, [col]])
                    else:
                        st.success("✅ No statistical anomalies found")

                    # Isolation Forest Anomalies
                    if anomaly_report['isolation_forest_anomalies']:
                        st.warning(f"Isolation Forest detected {len(anomaly_report['isolation_forest_anomalies'])} anomalies")
                        # Plot isolation forest anomalies in 2D
                        numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
                        if len(numeric_cols) >= 2:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plt.scatter(processed_data[numeric_cols[0]], processed_data[numeric_cols[1]], 
                                      alpha=0.5, label='Normal')
                            anomaly_data = processed_data.iloc[anomaly_report['isolation_forest_anomalies']]
                            plt.scatter(anomaly_data[numeric_cols[0]], anomaly_data[numeric_cols[1]], 
                                      color='red', label='Anomaly')
                            plt.xlabel(numeric_cols[0])
                            plt.ylabel(numeric_cols[1])
                            plt.title('Isolation Forest Anomalies')
                            plt.legend()
                            st.pyplot(fig)
                        st.dataframe(processed_data.loc[anomaly_report['isolation_forest_anomalies']])
                    else:
                        st.success("✅ No isolation forest anomalies found")

                    # Correlation Anomalies
                    if anomaly_report['correlation_anomalies']:
                        st.warning("Suspicious Correlations Found:")
                        for anomaly in anomaly_report['correlation_anomalies']:
                            st.write(f"- Strong correlation ({anomaly['correlation']:.2f}) between {anomaly['columns'][0]} and {anomaly['columns'][1]}")
                            # Plot correlation scatter
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plt.scatter(processed_data[anomaly['columns'][0]], 
                                      processed_data[anomaly['columns'][1]], alpha=0.5)
                            plt.xlabel(anomaly['columns'][0])
                            plt.ylabel(anomaly['columns'][1])
                            plt.title(f'Correlation between {anomaly["columns"][0]} and {anomaly["columns"][1]}')
                            st.pyplot(fig)
                    else:
                        st.success("✅ No suspicious correlations found")

                    # AI Assistant for Dataset Analysis
                    st.subheader("AI Dataset Assistant")
                    st.write("""
                    Ask questions about your dataset! For example:
                    - What are the main patterns in the data?
                    - Are there any concerning anomalies?
                    - What insights can you provide about the correlations?
                    - How can I improve data quality?
                    """)

                    # Update chat context with current dataset information
                    st.session_state.chat_context = {
                        'shape': processed_data.shape,
                        'columns': processed_data.columns.tolist(),
                        'dtypes': processed_data.dtypes.to_dict(),
                        'missing_values': processed_data.isnull().sum().to_dict(),
                        'numeric_summary': processed_data.describe().to_dict(),
                        'anomalies': {
                            'range_anomalies': validation_report['range_anomalies'],
                            'temporal_inconsistencies': len(validation_report['temporal_inconsistencies']),
                            'statistical_anomalies': anomaly_report['statistical_anomalies'],
                            'isolation_forest_anomalies': len(anomaly_report['isolation_forest_anomalies'])
                        }
                    }

                    # Display chat messages
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    # Chat input
                    if prompt := st.chat_input("Ask a question about your data"):
                        # Display user message
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        st.session_state.messages.append({"role": "user", "content": prompt})

                        # Generate and display assistant response
                        with st.chat_message("assistant"):
                            response = st.session_state.chat_assistant.generate_response(prompt, st.session_state.chat_context)
                            st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    # Clear chat button
                    if st.button("Clear Chat History"):
                        st.session_state.messages = []
                        st.session_state.chat_context = None
                        st.session_state.chat_assistant.clear_conversation()
                        st.rerun()

                    # Download processed data
                    st.subheader("Download Processed Data")
                    st.download_button(
                        label="Download processed dataset",
                        data=processed_data.to_csv(index=False),
                        file_name="processed_climate_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Unable to process the data. Please ensure it's in the correct format.")
            else:
                st.error("Failed to process the uploaded file. Please check the file format.")
        else:
            st.error("Error saving the file. Please try again.")

elif page == "Climate Data Download":
    st.title("Climate Data Download")
    st.write("Download climate data using the CDS API")
    
    if not st.session_state.climate_retriever.is_configured:
        st.error("CDS API is not configured!")
        st.markdown("""
        ### Setup Instructions
        1. Go to [Climate Data Store](https://cds.climate.copernicus.eu/)
        2. Create an account or log in
        3. Go to your profile page
        4. Copy your API key
        5. Edit the file `C:/Users/91709/.cdsapirc` `API-KEY-HERE` with your actual API key
        url: https://cds.climate.copernicus.eu/api/v2
        key: your-actual-api-key-here
        ```
        """)
    else:
        # Input form for CDS API parameters
        with st.form("climate_data_form"):
            dataset = st.text_input("Dataset Short Name", help="Enter the CDS dataset short name")
            target_file = st.text_input("Target File Name", help="Enter the name for the downloaded file (e.g., data.grib)")
            
            # Add request parameters (you can customize these based on your needs)
            st.subheader("Request Parameters")
            st.write("Enter your request parameters in JSON format:")
            request_params = st.text_area("Request Parameters", "{}")
            
            submit_button = st.form_submit_button("Download Data")
            
            if submit_button and dataset and target_file and request_params:
                try:
                    # Process the request parameters
                    import json
                    request_dict = json.loads(request_params)
                    
                    # Set the full path for the target file
                    full_target_path = os.path.join(UPLOAD_FOLDER, target_file)
                    
                    # Retrieve the climate data
                    with st.spinner("Downloading climate data..."):
                        downloaded_file = st.session_state.climate_retriever.retrieve_climate_data(
                            dataset,
                            request_dict,
                            full_target_path
                        )
                        
                    st.success(f"Data successfully downloaded to: {downloaded_file}")
                    
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in request parameters. Please check the format.")
                except Exception as e:
                    st.error(f"Error downloading data: {str(e)}")

elif page == "Developer Section":
    st.title("Development")
    st.write("Meet the developer of climate data validator")

    # Lead Profile
    st.header("")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Using a default image path -
        try:
            st.image("static/images/team_lead.jpg", use_container_width=True)
        except:
            st.image("./Profile Pics/IMG_20250111_224008.jpg", use_container_width=True)
    
    with col2:
        st.subheader("Gitartha Talukdar")  
        st.write("""
        Role :  Project Lead & Senior Developer
        
        Expertise :
        - Artifilcial Intelligence & Machine Learning
        - Data Analysis / Cybersecurity / Data Science
        - Full Stack Development
        
        Contact :   gitarthatalukdar0@gmail.com
        """)


    # Project Information
    st.header("Project Overview")
    st.write("""
    This Climate Data Validator project was developed as part of our commitment to 
    improving climate data analysis and validation. Ithas focused on creating 
    a user-friendly interface while maintaining robust data processing capabilities.
    """)

    # Contact Information
    st.header("Contact Us")
    st.write("""
    For any queries or suggestions, please reach out to us:
    - 📧 Email: gitarthatalukdar0@gmail.com
    - 🌐 GitHub: https://github.com/Gitarthaaa
    """)
