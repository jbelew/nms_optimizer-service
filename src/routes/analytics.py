import time
import datetime
from flask import Blueprint, jsonify, request, current_app
from google.cloud import bigquery
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Filter,
    FilterExpression,
    Metric,
    OrderBy,
    RunReportRequest,
)

from ..analytics_data import ga4_data_client, bq_client, GA_PROPERTY_ID
from ..analytics import send_analytics_event

analytics_bp = Blueprint("analytics", __name__)


# --- Caching Support ---
class SimpleCache:
    def __init__(self, ttl_seconds=900):  # 15 minutes default
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, key):
        entry = self.cache.get(key)
        if entry and (time.time() - entry["timestamp"]) < self.ttl:
            return entry["data"]
        return None

    def set(self, key, data):
        self.cache[key] = {"data": data, "timestamp": time.time()}


perf_cache = SimpleCache()
# --- End Caching Support ---


def parse_ga4_date(date_str: str) -> datetime.date:
    """Parses GA4 relative date strings or YYYY-MM-DD into a date object."""
    today = datetime.date.today()
    if date_str == "today":
        return today
    if date_str == "yesterday":
        return today - datetime.timedelta(days=1)
    if date_str.endswith("daysAgo"):
        try:
            days = int(date_str.replace("daysAgo", ""))
            return today - datetime.timedelta(days=days)
        except ValueError:
            pass

    # Fallback to YYYY-MM-DD or 30 days ago
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return today - datetime.timedelta(days=30)


@analytics_bp.route("/analytics/popular_data", methods=["GET"])
def get_popular_analytics_data():
    """Fetches and returns popular optimization data from Google Analytics."""
    if not ga4_data_client:
        return jsonify({"error": "Google Analytics Data API client not initialized."}), 500

    try:
        start_date = request.args.get("start_date", "30daysAgo")
        end_date = request.args.get("end_date", "today")

        request_body = RunReportRequest(
            property=f"properties/{GA_PROPERTY_ID}",
            date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
            dimensions=[
                Dimension(name="customEvent:platform"),
                Dimension(name="customEvent:tech"),
                Dimension(name="customEvent:supercharged"),
            ],
            dimension_filter=FilterExpression(
                filter=Filter(
                    field_name="eventName",
                    string_filter=Filter.StringFilter(value="optimize_tech"),
                )
            ),
            metrics=[Metric(name="eventCount")],
            order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="eventCount"), desc=True)],
        )

        response = ga4_data_client.run_report(request_body)

        popular_data = []
        for row in response.rows:
            popular_data.append(
                {
                    "event_name": "optimize_tech",
                    "ship_type": row.dimension_values[0].value,
                    "technology": row.dimension_values[1].value,
                    "supercharged": row.dimension_values[2].value,
                    "total_events": int(row.metric_values[0].value),
                }
            )

        return jsonify(popular_data)

    except Exception as e:
        current_app.logger.error(f"Error fetching Google Analytics data: {e}")
        return jsonify({"error": f"Failed to fetch analytics data: {e}"}), 500


@analytics_bp.route("/analytics/performance_data", methods=["GET"])
def get_performance_analytics_data():
    """Fetches aggregate performance metrics from BigQuery with GA4 fallback."""
    start_date_param = request.args.get("start_date", "30daysAgo")
    end_date_param = request.args.get("end_date", "today")

    cache_key = f"perf_{start_date_param}_{end_date_param}"

    # Check cache first
    cached_data = perf_cache.get(cache_key)
    if cached_data:
        return jsonify(cached_data)

    # Try BigQuery first for true percentiles
    if bq_client:
        try:
            start_date = parse_ga4_date(start_date_param)
            end_date = parse_ga4_date(end_date_param)

            query = """
                WITH raw_source AS (
                  SELECT
                    event_timestamp,
                    user_pseudo_id,
                    event_params,
                    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'metric_name') as m_name,
                    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'label') as m_id,
                    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'app_version') as v,
                    (SELECT COALESCE(value.int_value, value.double_value, SAFE_CAST(value.string_value AS FLOAT64))
                     FROM UNNEST(event_params) WHERE key = 'value') as val
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
                    WHERE _TABLE_SUFFIX BETWEEN
                        FORMAT_DATE('%Y%m%d', @start_date) AND
                        FORMAT_DATE('%Y%m%d', @end_date)
                      AND event_name = 'performance_metric'
                  )
                ),
                deduped_vitals AS (
                  -- Deduplicate by metric ID (from web-vitals) to prevent doubling between tables
                  SELECT
                    ANY_VALUE(m_name) as m_name,
                    ANY_VALUE(v) as v,
                    ANY_VALUE(event_timestamp) as ts,
                    SUM(val) as total_val -- Sum deltas for the same ID
                  FROM raw_source
                  WHERE m_name IS NOT NULL AND m_name != 'TBT'
                  GROUP BY COALESCE(m_id, CAST(event_timestamp AS STRING) || user_pseudo_id || m_name)
                ),
                hourly_stats AS (
                  SELECT
                    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(ts), HOUR) as hr,
                    m_name as metric_name,
                    APPROX_TOP_COUNT(v, 1)[OFFSET(0)].value as app_version,
                    APPROX_QUANTILES(total_val, 100)[OFFSET(50)] as p50_val,
                    APPROX_QUANTILES(total_val, 100)[OFFSET(75)] as p75_val,
                    APPROX_QUANTILES(total_val, 100)[OFFSET(90)] as p90_val
                  FROM deduped_vitals
                  GROUP BY 1, 2
                  HAVING COUNT(*) >= 5 -- Still require 5 samples for statistical significance
                )
                SELECT
                  UNIX_MILLIS(s.hr) as timestamp,
                  s.metric_name,
                  s.app_version,
                  s.p50_val,
                  s.p75_val as average_value,
                  s.p90_val
                FROM hourly_stats s
                WHERE s.hr < TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), HOUR)
                ORDER BY 1 ASC
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
                    bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
                ]
            )
            query_job = bq_client.query(query, job_config=job_config)
            results = query_job.result()

            performance_data = []
            for row in results:
                if row.metric_name and row.average_value is not None:
                    performance_data.append(
                        {
                            "timestamp": row.timestamp,
                            "metric_name": row.metric_name,
                            "app_version": row.app_version or "unknown",
                            "p50": float(row.p50_val) if row.p50_val is not None else None,
                            "p75": float(row.average_value),
                            "p90": float(row.p90_val) if row.p90_val is not None else None,
                            "average_value": float(row.average_value),
                        }
                    )

            if performance_data:
                perf_cache.set(cache_key, performance_data)
                return jsonify(performance_data)

        except Exception as bq_err:
            current_app.logger.warning(f"BigQuery performance query failed, falling back to GA4: {bq_err}")

    # Fallback to GA4 Reporting API (Averages only)
    if not ga4_data_client:
        return jsonify({"error": "No analytics clients available."}), 500

    try:
        request_body = RunReportRequest(
            property=f"properties/{GA_PROPERTY_ID}",
            date_ranges=[DateRange(start_date=start_date_param, end_date=end_date_param)],
            dimensions=[
                Dimension(name="dateHour"),
                Dimension(name="customEvent:metric_name"),
            ],
            dimension_filter=FilterExpression(
                filter=Filter(
                    field_name="eventName",
                    string_filter=Filter.StringFilter(value="performance_metric"),
                )
            ),
            metrics=[
                Metric(name="averageCustomEvent:value"),
            ],
            order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name="dateHour"), desc=False)],
        )

        response = ga4_data_client.run_report(request_body)

        performance_data = []
        for row in response.rows:
            date_hour_str = row.dimension_values[0].value
            # Convert YYYYMMDDHH to Unix timestamp (milliseconds)
            dt = datetime.datetime.strptime(date_hour_str, "%Y%m%d%H")
            timestamp = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)

            performance_data.append(
                {
                    "timestamp": timestamp,
                    "metric_name": row.dimension_values[1].value,
                    "app_version": "unknown",
                    "p50": None,
                    "p75": float(row.metric_values[0].value),
                    "p90": None,
                    "average_value": float(row.metric_values[0].value),
                }
            )

        if performance_data:
            perf_cache.set(cache_key, performance_data)

        return jsonify(performance_data)

    except Exception as e:
        current_app.logger.error(f"Error fetching performance analytics data: {e}")
        return jsonify({"error": f"Failed to fetch performance data: {e}"}), 500


@analytics_bp.route("/api/events", methods=["POST"])
def track_event():
    """Receive and relay analytics events to GA4."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        client_id = data.get("clientId")
        event_name = data.get("eventName")
        params = data.get("params", {})
        user_id = data.get("userId")

        # Validate required fields
        if not client_id or not event_name:
            return jsonify({"error": "Missing required fields: clientId, eventName"}), 400

        # Send event to GA4
        success = send_analytics_event(
            event_name=event_name,
            client_id=client_id,
            params=params,
            user_id=user_id,
        )

        if success:
            return jsonify({"status": "ok"}), 200
        else:
            return jsonify({"error": "Failed to send event to GA4"}), 500

    except Exception as e:
        current_app.logger.error(f"Error in analytics endpoint: {e}")
        return jsonify({"error": str(e)}), 500
