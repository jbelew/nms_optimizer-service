import datetime
from google.cloud import bigquery
from src.analytics_data import initialize_clients

ga4_data_client, bq_client = initialize_clients()

if not bq_client:
    print("No BQ client")
    exit(1)

start_date = datetime.date.today() - datetime.timedelta(days=30)
end_date = datetime.date.today()

query = """
    SELECT
        (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'metric_name') as m_name,
        COUNT(*) as count
    FROM `cosmic-inkwell-467922-v5.analytics_484727815.events_*`
    WHERE _TABLE_SUFFIX BETWEEN
        FORMAT_DATE('%Y%m%d', @start_date) AND
        FORMAT_DATE('%Y%m%d', @end_date)
      AND event_name = 'performance_metric'
    GROUP BY 1
"""

try:
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    query_job = bq_client.query(query, job_config=job_config)
    results = query_job.result()
    for row in results:
        print(f"{row.m_name}: {row.count}")
except Exception as e:
    print(f"Error: {e}")
