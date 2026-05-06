from src.analytics_data import initialize_clients

ga4_data_client, bq_client = initialize_clients()

query = """
    SELECT event_name, COUNT(*) as count
    FROM `cosmic-inkwell-467922-v5.analytics_484727815.events_intraday_*`
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 10
"""

try:
    query_job = bq_client.query(query)
    results = query_job.result()
    for row in results:
        print(f"{row.event_name}: {row.count}")
except Exception as e:
    print(f"Error: {e}")
