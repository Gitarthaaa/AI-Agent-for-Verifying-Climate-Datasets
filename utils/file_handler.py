import os
import pandas as pd
from docx import Document

def save_file(file, upload_folder):
    filepath = os.path.join(upload_folder, file.name)
    with open(filepath, "wb") as f:
        f.write(file.getbuffer())
    return filepath

def extract_data(filepath):
    if filepath.endswith('.csv'):
        try:
            # First try with default UTF-8 encoding
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try with different encodings
                df = pd.read_csv(filepath, encoding='latin1')

            # Basic CSV validation
            if df.empty:
                print("Warning: CSV file is empty")
                return None

            # Handle date columns automatically
            for col in df.columns:
                # Try to convert to datetime if column name suggests it's a date
                if any(date_hint in col.lower() for date_hint in ['date', 'time', 'year', 'month']):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception as e:
                        print(f"Warning: Could not convert {col} to datetime: {e}")

            # Check for common data quality issues
            null_counts = df.isnull().sum()
            if null_counts.any():
                print("Warning: Missing values detected in columns:", 
                      ", ".join(f"{col} ({count} nulls)" for col, count in null_counts[null_counts > 0].items()))

            return df

        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty")
            return None
        except pd.errors.ParserError as e:
            # Try with different delimiters if standard comma fails
            for delimiter in [';', '\t', '|']:
                try:
                    df = pd.read_csv(filepath, sep=delimiter)
                    print(f"Successfully read CSV with delimiter: {delimiter}")
                    return df
                except:
                    continue
            print(f"Error parsing CSV: {e}")
            return None
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None
    elif filepath.endswith('.docx'):
        try:
            doc = Document(filepath)
            return [para.text for para in doc.paragraphs if para.text.strip()]
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return None
    else:
        return None
