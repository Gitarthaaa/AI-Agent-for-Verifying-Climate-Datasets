import streamlit as st

class ChatAssistant:
    def __init__(self):
        """Initialize the chat assistant."""
        self.conversation_history = []
        self.is_configured = True

    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def generate_response(self, user_input: str, dataset_context: dict = None) -> str:
        """
        Generate a response to the user's question about the dataset.
        
        Args:
            user_input (str): The user's question
            dataset_context (dict): Context about the dataset including shape, columns, etc.
            
        Returns:
            str: The assistant's response
        """
        try:
            # Convert user input to lowercase for easier matching
            user_input_lower = user_input.lower()
            
            # Prepare dataset information
            dataset_info = ""
            if dataset_context:
                num_rows, num_cols = dataset_context['shape']
                columns = dataset_context['columns']
                missing_values = dataset_context['missing_values']
                anomalies = dataset_context.get('anomalies', {})
                
                # Basic dataset statistics
                total_anomalies = (
                    len(anomalies.get('range_anomalies', {})) +
                    anomalies.get('temporal_inconsistencies', 0) +
                    len(anomalies.get('statistical_anomalies', {})) +
                    anomalies.get('isolation_forest_anomalies', 0)
                )

            # Pattern matching for different types of questions
            if 'pattern' in user_input_lower or 'trend' in user_input_lower:
                response = f"Based on the analysis of your dataset ({num_rows} rows, {num_cols} columns), "
                if total_anomalies > 0:
                    response += f"I found {total_anomalies} potential anomalies that might affect the patterns. "
                response += "To identify patterns, look at the Time Series Analysis section which shows trends over time, "
                response += "and the Distribution Analysis which shows the shape of your data distribution."

            elif 'anomal' in user_input_lower or 'outlier' in user_input_lower:
                response = f"I detected several types of anomalies in your dataset:\n"
                if anomalies.get('range_anomalies'):
                    response += f"- {len(anomalies['range_anomalies'])} values outside expected ranges\n"
                if anomalies.get('temporal_inconsistencies'):
                    response += f"- {anomalies['temporal_inconsistencies']} time gaps or inconsistencies\n"
                if anomalies.get('statistical_anomalies'):
                    response += f"- {len(anomalies['statistical_anomalies'])} statistical outliers\n"
                if anomalies.get('isolation_forest_anomalies'):
                    response += f"- {anomalies['isolation_forest_anomalies']} anomalies detected by Isolation Forest\n"
                response += "\nCheck the Advanced Anomaly Detection section for visualizations of these anomalies."

            elif 'correlation' in user_input_lower or 'relationship' in user_input_lower:
                response = "You can find correlation analysis in the Correlation Analysis section, "
                response += "which shows a heatmap of relationships between numeric variables. "
                response += "Look for dark red (strong positive) or dark blue (strong negative) colors "
                response += "in the heatmap to identify significant correlations."

            elif 'quality' in user_input_lower or 'improve' in user_input_lower:
                response = "Here are some suggestions to improve data quality:\n"
                if any(missing_values.values()):
                    response += "1. Handle missing values - you have some gaps in your data\n"
                if total_anomalies > 0:
                    response += "2. Investigate and validate the detected anomalies\n"
                response += "3. Check for consistent data formats and units\n"
                response += "4. Consider adding more contextual variables if available\n"
                response += "5. Implement regular data validation checks"

            elif 'missing' in user_input_lower or 'gap' in user_input_lower:
                response = "Regarding missing values in your dataset:\n"
                for col, count in missing_values.items():
                    if count > 0:
                        response += f"- {col}: {count} missing values\n"
                if not any(missing_values.values()):
                    response += "Good news! Your dataset has no missing values."

            else:
                response = f"Your dataset contains {num_rows} rows and {num_cols} columns. "
                response += f"The available columns are: {', '.join(columns)}. "
                response += "\nYou can ask me about:\n"
                response += "- Patterns and trends in the data\n"
                response += "- Anomalies and outliers\n"
                response += "- Correlations between variables\n"
                response += "- Data quality and improvements\n"
                response += "- Missing values and gaps"

            # Store conversation
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            return f"I encountered an error while analyzing your question: {str(e)}. Please try asking in a different way or check if you have uploaded a dataset."
