
import os
from google.cloud import bigquery
from google.oauth2 import service_account

# Path to your credentials
KEY_FILE = "src/cosmic-inkwell-467922-v5-85707f3bcc80.json"
PROJECT_ID = "cosmic-inkwell-467922-v5"
DATASET_ID = "analytics_484727815"

def test_connection():
    print(f"--- Testing BigQuery Connection for {PROJECT_ID} ---")
    try:
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials = service_account.Credentials.from_service_account_file(KEY_FILE, scopes=scopes)
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        
        # Test 1: List datasets to verify basic access
        datasets = list(client.list_datasets())
        print(f"Successfully connected. Found {len(datasets)} datasets.")
        
        # Test 2: Run the p75 query (Hourly)
        print("\n--- Running p75 Hourly Performance Query ---")
        query = f"""
            SELECT
              FORMAT_TIMESTAMP('%Y%m%d%H', TIMESTAMP_MICROS(event_timestamp)) as date,
              (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'metric_name') as metric_name,
              APPROX_QUANTILES(
                COALESCE(
                    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'value'),
                    (SELECT value.double_value FROM UNNEST(event_params) WHERE key = 'value'),
                    SAFE_CAST((SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'value') AS FLOAT64)
                ), 100)[OFFSET(75)] as p75_value
            FROM
              (
                SELECT event_timestamp, event_params, event_name FROM `{PROJECT_ID}.{DATASET_ID}.events_*`
                WHERE _TABLE_SUFFIX >= FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY))
                UNION ALL
                SELECT event_timestamp, event_params, event_name FROM `{PROJECT_ID}.{DATASET_ID}.events_intraday_*`
                WHERE _TABLE_SUFFIX >= FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY))
              )
            WHERE
              event_name = 'performance_metric'
            GROUP BY 1, 2
            ORDER BY 1 ASC
            LIMIT 20
        """
        
        query_job = client.query(query)
        results = query_job.result()
        
        print(f"{'Date':<10} | {'Metric':<10} | {'p75 Value':<10}")
        print("-" * 35)
        count = 0
        for row in results:
            count += 1
            print(f"{row.date:<10} | {row.metric_name:<10} | {row.p75_value:<10.2f}")
        
        if count == 0:
            print("Query returned 0 rows. (Maybe streaming hasn't populated data yet?)")
            
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    test_connection()
