import os
import logging
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.cloud import bigquery
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# --- Google Analytics 4 (GA4) Configuration ---
GA_PROPERTY_ID = "484727815"


def initialize_clients():
    """Initializes the Google Analytics and BigQuery clients.

    It attempts to load credentials first from the environment variable
    `GOOGLE_APPLICATION_CREDENTIALS_JSON` (for services like Heroku) and
    falls back to a local JSON key file for development.

    Returns:
        tuple: (ga4_data_client, bq_client) instances, or (None, None) if initialization fails.
    """
    try:
        # Try to get credentials from environment variable first (Heroku)
        gcp_key_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        scopes = [
            "https://www.googleapis.com/auth/analytics.readonly",
            "https://www.googleapis.com/auth/cloud-platform",
        ]

        if gcp_key_json:
            import json

            credentials_info = json.loads(gcp_key_json)
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=scopes,
            )
        else:
            # Fallback to file-based credentials (local development)
            GA_KEY_FILE_PATH = os.path.join(os.path.dirname(__file__), "cosmic-inkwell-467922-v5-85707f3bcc80.json")
            if not os.path.exists(GA_KEY_FILE_PATH):
                logger.warning(f"GA key file not found at {GA_KEY_FILE_PATH}")
                return None, None

            credentials = service_account.Credentials.from_service_account_file(
                GA_KEY_FILE_PATH,
                scopes=scopes,
            )

        ga_data_client = BetaAnalyticsDataClient(credentials=credentials)
        bq_client = bigquery.Client(credentials=credentials, project="cosmic-inkwell-467922-v5")

        return ga_data_client, bq_client
    except Exception as e:
        logger.error(f"Error initializing Google Cloud clients: {e}")
        return None, None


# Initialize singleton instances
ga4_data_client, bq_client = initialize_clients()
