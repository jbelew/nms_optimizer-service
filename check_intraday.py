from src.analytics_data import initialize_clients

ga4_data_client, bq_client = initialize_clients()

if not bq_client:
    print("No BQ client")
    exit(1)

query = """
    SELECT
        (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'metric_name') as m_name,
        COUNT(*) as count
    FROM `cosmic-inkwell-467922-v5.analytics_484727815.events_intraday_*`
    WHERE _TABLE_SUFFIX = FORMAT_DATE('%Y%m%d', CURRENT_DATE())
      AND event_name = 'performance_metric'
    GROUP BY 1
"""

try:
    query_job = bq_client.query(query)
    results = query_job.result()
    for row in results:
        print(f"{row.m_name}: {row.count}")
except Exception as e:
    print(f"Error: {e}")
