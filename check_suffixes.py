from google.cloud import bigquery
from src.analytics_data import initialize_clients

ga4_data_client, bq_client = initialize_clients()

query = """
    SELECT _TABLE_SUFFIX as suffix, COUNT(*) as count
    FROM `cosmic-inkwell-467922-v5.analytics_484727815.events_intraday_*`
    GROUP BY 1
    ORDER BY 1 DESC
"""

try:
    query_job = bq_client.query(query)
    results = query_job.result()
    for row in results:
        print(f"{row.suffix}: {row.count}")
except Exception as e:
    print(f"Error: {e}")
