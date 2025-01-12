import cdsapi
import os
from typing import Dict, Any

class ClimateDataRetriever:
    def __init__(self):
        """Initialize the CDS API client."""
        try:
            self.client = cdsapi.Client()
            self.is_configured = True
        except Exception as e:
            self.client = None
            self.is_configured = False
            self.error_message = str(e)

    def retrieve_climate_data(
        self,
        dataset: str,
        request_params: Dict[str, Any],
        target_file: str
    ) -> str:
        """
        Retrieve climate data using the CDS API.
        
        Args:
            dataset (str): The dataset short name from CDS
            request_params (dict): The selection parameters for the data request
            target_file (str): The target file path to save the retrieved data
            
        Returns:
            str: Path to the downloaded file if successful
        
        Raises:
            Exception: If there's an error during data retrieval or if API is not configured
        """
        if not self.is_configured:
            raise Exception(
                "CDS API is not properly configured. Please set up your API key in ~/.cdsapirc\n"
                "1. Go to https://cds.climate.copernicus.eu/\n"
                "2. Create an account or log in\n"
                "3. Go to your profile page\n"
                "4. Copy your API key\n"
                "5. Edit ~/.cdsapirc and replace YOUR-API-KEY-HERE with your actual key"
            )
            
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            
            # Retrieve the data
            self.client.retrieve(
                dataset,
                request_params,
                target_file
            )
            
            return target_file
            
        except Exception as e:
            raise Exception(f"Error retrieving climate data: {str(e)}")
