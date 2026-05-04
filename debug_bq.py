import datetime
from google.cloud import bigquery
from src.analytics_data import initialize_clients

ga4_data_client, bq_client = initialize_clients()

if not bq_client:
    print("No BQ client")
    exit(1)

start_date = datetime.date.today() - datetime.timedelta(days=7)
end_date = datetime.date.today()

query = """
    WITH raw_source AS (
      SELECT
        event_timestamp,
        user_pseudo_id,
        event_params
      FROM (
        SELECT event_timestamp, user_pseudo_id, event_params
        FROM `cosmic-inkwell-467922-v5.analytics_484727815.events_*`
        WHERE _TABLE_SUFFIX BETWEEN
            FORMAT_DATE('%Y%m%d', @start_date) AND
            FORMAT_DATE('%Y%m%d', @end_date)
          AND event_name = 'performance_metric'

        UNION ALL

        SELECT event_timestamp, user_pseudo_id, event_params
        FROM `cosmic-inkwell-467922-v5.analytics_484727815.events_intraday_*`
        WHERE _TABLE_SUFFIX = FORMAT_DATE('%Y%m%d', CURRENT_DATE())
          AND event_name = 'performance_metric'
      )
    ),
    deduped_metrics AS (
      SELECT DISTINCT
        event_timestamp,
        user_pseudo_id,
        (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'metric_name') as m_name,
        (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'label') as m_id,
        (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'app_version') as v,
        (SELECT COALESCE(value.int_value, value.double_value, SAFE_CAST(value.string_value AS FLOAT64))
         FROM UNNEST(event_params) WHERE key = 'value') as val
      FROM raw_source
    ),
    per_metric_totals AS (
      SELECT
        ANY_VALUE(m_name) as m_name,
        ANY_VALUE(v) as v,
        MIN(event_timestamp) as first_ts,
        SUM(val) as total_val
      FROM deduped_metrics
      WHERE m_name IS NOT NULL AND m_name != 'TBT'
      GROUP BY COALESCE(m_id, CAST(event_timestamp AS STRING) || user_pseudo_id || m_name)
    ),
    hourly_stats AS (
      SELECT
        TIMESTAMP_TRUNC(TIMESTAMP_MICROS(first_ts), HOUR) as hr,
        m_name as metric_name,
        APPROX_TOP_COUNT(v, 1)[OFFSET(0)].value as app_version,
        APPROX_QUANTILES(total_val, 100)[OFFSET(50)] as p50_val,
        APPROX_QUANTILES(total_val, 100)[OFFSET(75)] as p75_val,
        APPROX_QUANTILES(total_val, 100)[OFFSET(90)] as p90_val
      FROM per_metric_totals
      GROUP BY 1, 2
      HAVING COUNT(*) >= 5
    ),
    complete_hours AS (
      SELECT hr FROM hourly_stats
      GROUP BY hr
      HAVING COUNT(DISTINCT metric_name) = 5
    )
    SELECT
      UNIX_MILLIS(s.hr) as timestamp,
      s.metric_name,
      s.app_version,
      s.p50_val,
      s.p75_val as average_value,
      s.p90_val
    FROM hourly_stats s
    INNER JOIN complete_hours c ON s.hr = c.hr
    WHERE s.hr < TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), HOUR)
    ORDER BY 1 ASC
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
    print(f"Success! Found {len(list(results))} rows.")
except Exception as e:
    print(f"Error: {e}")
