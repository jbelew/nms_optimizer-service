import time
import datetime
import threading
from collections import OrderedDict
from typing import Any
from flask import Blueprint, Response, current_app, jsonify, request
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
# NOTE: This is an in-process cache. In multi-worker deployments (e.g., Gunicorn),
# each worker keeps its own copy. Use Redis/Memcached via Flask-Caching for
# shared state across workers.
class SimpleCache:
    """Thread-safe bounded TTL cache for in-process analytics responses."""

    def __init__(self, ttl_seconds: int = 900, max_size: int = 128) -> None:
        """Initialize the cache with a TTL and maximum entry count.

        Args:
            ttl_seconds: Time-to-live per cached entry in seconds (default 900).
            max_size: Maximum number of keys retained before LRU eviction.
        """
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Return cached data for ``key`` if present and unexpired, else ``None``.

        Args:
            key: Cache lookup key.

        Returns:
            Any | None: Cached payload when valid, or ``None`` on miss/expiry.
        """
        with self._lock:
            entry = self.cache.get(key)
            if entry and (time.time() - entry["timestamp"]) < self.ttl:
                self.cache.move_to_end(key)
                return entry["data"]
            if entry:
                # Expired; drop it
                self.cache.pop(key, None)
            return None

    def set(self, key: str, data: Any) -> None:
        """Store ``data`` under ``key`` and evict oldest entries beyond ``max_size``.

        Args:
            key: Cache storage key.
            data: Serializable response payload to cache.
        """
        with self._lock:
            self.cache[key] = {"data": data, "timestamp": time.time()}
            self.cache.move_to_end(key)
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)


perf_cache = SimpleCache()
popular_cache = SimpleCache()
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
def get_popular_analytics_data() -> tuple[Response, int] | Response:
    """Fetches and returns popular optimization data from BigQuery with GA4 fallback.

    Returns:
        tuple[flask.Response, int] | flask.Response: JSON response containing popular
        technology optimization counts per ship type and supercharged status.
    """
    start_date_param = request.args.get("start_date", "30daysAgo")
    end_date_param = request.args.get("end_date", "today")

    # Include current hour in cache key to ensure fresh daily/hourly aggregates
    current_hour = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H")
    cache_key = f"popular_{start_date_param}_{end_date_param}_{current_hour}"

    # Check cache first
    cached_data = popular_cache.get(cache_key)
    if cached_data:
        return jsonify(cached_data)

    # Parse dates once up-front so both BigQuery and GA4 fallback use validated values.
    start_date = parse_ga4_date(start_date_param)
    end_date = parse_ga4_date(end_date_param)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    start_suffix = start_date.strftime("%Y%m%d")
    end_suffix = end_date.strftime("%Y%m%d")

    # Try BigQuery first for fast, reliable data retrieval
    if bq_client:
        try:
            query = f"""
                WITH raw_source_unioned AS (
                  SELECT
                    event_timestamp,
                    user_pseudo_id,
                    event_params
                  FROM (
                    SELECT event_timestamp, user_pseudo_id, event_params
                    FROM `cosmic-inkwell-467922-v5.analytics_484727815.events_*`
                    WHERE _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
                      AND event_name = 'optimize_tech'

                    UNION ALL

                    SELECT event_timestamp, user_pseudo_id, event_params
                    FROM `cosmic-inkwell-467922-v5.analytics_484727815.events_intraday_*`
                    WHERE _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
                      AND event_name = 'optimize_tech'
                  )
                ),
                extracted_events AS (
                  SELECT
                    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'platform') AS ship_type,
                    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'tech') AS technology,
                    COALESCE(
                      (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'supercharged'),
                      (SELECT IF(value.int_value = 1, 'true', IF(value.int_value = 0, 'false', NULL)) FROM UNNEST(event_params) WHERE key = 'supercharged'),
                      'false'
                    ) AS supercharged
                  FROM raw_source_unioned
                )
                SELECT
                  ship_type,
                  technology,
                  supercharged,
                  COUNT(1) AS total_events
                FROM extracted_events
                WHERE ship_type IS NOT NULL AND technology IS NOT NULL
                GROUP BY 1, 2, 3
                ORDER BY total_events DESC
            """

            query_job = bq_client.query(query)
            results = query_job.result()

            popular_data = []
            for row in results:
                if row.ship_type and row.technology:
                    popular_data.append(
                        {
                            "event_name": "optimize_tech",
                            "ship_type": row.ship_type,
                            "technology": row.technology,
                            "supercharged": row.supercharged or "false",
                            "total_events": int(row.total_events),
                        }
                    )

            if popular_data:
                popular_cache.set(cache_key, popular_data)
                return jsonify(popular_data)

        except Exception as bq_err:
            current_app.logger.warning(f"BigQuery popular query failed, falling back to GA4: {bq_err}")

    # Fallback to GA4 Data API (REST)
    if not ga4_data_client:
        return jsonify({"error": "No analytics clients available."}), 500

    try:
        request_body = RunReportRequest(
            property=f"properties/{GA_PROPERTY_ID}",
            date_ranges=[DateRange(start_date=start_date_str, end_date=end_date_str)],
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

        if popular_data:
            popular_cache.set(cache_key, popular_data)
        return jsonify(popular_data)

    except Exception as e:
        current_app.logger.error(f"Error fetching Google Analytics data: {e}")
        return jsonify({"error": f"Failed to fetch analytics data: {e}"}), 500


@analytics_bp.route("/analytics/performance_data", methods=["GET"])
def get_performance_analytics_data() -> tuple[Response, int] | Response:
    """Fetches aggregate production Web Vitals metrics from BigQuery with GA4 fallback.

    Returns:
        tuple[Response, int] | Response: JSON response containing hourly
        p50, p75, and p90 Core Web Vitals percentiles per metric and app version.
    """
    start_date_param = request.args.get("start_date", "30daysAgo")
    end_date_param = request.args.get("end_date", "today")

    # Include current hour in cache key to ensure we don't serve stale data
    # that excludes a newly completed hour.
    current_hour = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H")
    cache_key = f"perf_{start_date_param}_{end_date_param}_{current_hour}"

    # Check cache first
    cached_data = perf_cache.get(cache_key)
    if cached_data:
        return jsonify(cached_data)

    # Parse dates once up-front so both BigQuery and GA4 fallback use validated values.
    start_date = parse_ga4_date(start_date_param)
    end_date = parse_ga4_date(end_date_param)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Try BigQuery first for true percentiles
    if bq_client:
        try:
            # NOTE: BigQuery does not allow query parameters inside `_TABLE_SUFFIX`
            # filters. The dates are derived from `parse_ga4_date`, which always
            # returns a `datetime.date`, so f-string interpolation of the
            # `YYYYMMDD` suffix is safe (no SQL-injection surface).
            start_suffix = start_date.strftime("%Y%m%d")
            end_suffix = end_date.strftime("%Y%m%d")

            query = f"""
                WITH raw_source_unioned AS (
                  SELECT
                    event_timestamp,
                    user_pseudo_id,
                    event_params,
                    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS page_location
                  FROM (
                    SELECT event_timestamp, user_pseudo_id, device.web_info.hostname AS hostname, event_params
                    FROM `cosmic-inkwell-467922-v5.analytics_484727815.events_*`
                    WHERE _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
                      AND event_name = 'performance_metric'

                    UNION ALL

                    SELECT event_timestamp, user_pseudo_id, device.web_info.hostname AS hostname, event_params
                    FROM `cosmic-inkwell-467922-v5.analytics_484727815.events_intraday_*`
                    WHERE _TABLE_SUFFIX BETWEEN '{start_suffix}' AND '{end_suffix}'
                      AND event_name = 'performance_metric'
                  )
                  WHERE LOWER(COALESCE(hostname, '')) IN ('', 'nms-optimizer.app', 'www.nms-optimizer.app')
                ),
                raw_source AS (
                  -- Deduplicate exact row duplicates from table overlap (daily vs intraday)
                  SELECT DISTINCT
                    event_timestamp,
                    user_pseudo_id,
                    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'metric_name') AS metric_name,
                    COALESCE(
                      (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'metric_id'),
                      (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'label')
                    ) AS metric_id,
                    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'app_version') AS app_version,
                    (SELECT COALESCE(value.int_value, value.double_value, SAFE_CAST(value.string_value AS FLOAT64))
                     FROM UNNEST(event_params) WHERE key = 'value') AS event_value
                  FROM raw_source_unioned
                  WHERE page_location IS NULL
                    OR page_location = ''
                    OR NOT REGEXP_CONTAINS(
                      LOWER(page_location),
                      r'^https?://(localhost|127\\.0\\.0\\.1|([^/?#:]+\\.)?pages\\.dev|([^/?#:]+\\.)?herokuapp\\.com)([:/?#]|$)'
                    )
                ),
                deduped_vitals AS (
                  -- Partition explicitly by (user_pseudo_id, metric_name, metric_instance_key) and bucket
                  -- deterministically by MAX(event_timestamp). Sum incremental deltas for CLS and
                  -- legacy pre-v8.4.3 INP; resolve v8.4.3+ INP and single-fire Web Vitals
                  -- (TTFB, FCP, LCP) to the latest chronological snapshot.
                  SELECT
                    metric_name,
                    app_version,
                    latest_timestamp,
                    CASE
                      WHEN metric_name = 'CLS' THEN delta_sum
                      WHEN metric_name = 'INP' AND REGEXP_CONTAINS(
                        COALESCE(app_version, ''),
                        r'^[vV]?([0-7]\\.\\d+\\.\\d+|8\\.[0-3]\\.\\d+|8\\.4\\.[0-2])($|[^0-9.])'
                      ) THEN delta_sum
                      ELSE latest_value
                    END AS metric_value
                  FROM (
                    SELECT
                      metric_name,
                      ARRAY_AGG(app_version IGNORE NULLS ORDER BY event_timestamp DESC LIMIT 1)[SAFE_OFFSET(0)] AS app_version,
                      MAX(event_timestamp) AS latest_timestamp,
                      SUM(event_value) AS delta_sum,
                      ARRAY_AGG(event_value IGNORE NULLS ORDER BY event_timestamp DESC LIMIT 1)[SAFE_OFFSET(0)] AS latest_value
                    FROM raw_source
                    WHERE metric_name IS NOT NULL AND metric_name != 'TBT' AND event_value IS NOT NULL
                    GROUP BY
                      user_pseudo_id,
                      metric_name,
                      IF(metric_id IS NOT NULL AND REGEXP_CONTAINS(metric_id, r'^v\\d+-\\d+-\\d+$'), metric_id, CAST(event_timestamp AS STRING))
                  )
                ),
                hourly_stats AS (
                  SELECT
                    TIMESTAMP_TRUNC(TIMESTAMP_MICROS(latest_timestamp), HOUR) AS hour_bucket,
                    metric_name,
                    APPROX_TOP_COUNT(app_version, 1)[SAFE_OFFSET(0)].value AS app_version,
                    APPROX_QUANTILES(metric_value, 100)[OFFSET(50)] AS p50_val,
                    APPROX_QUANTILES(metric_value, 100)[OFFSET(75)] AS p75_val,
                    APPROX_QUANTILES(metric_value, 100)[OFFSET(90)] AS p90_val
                  FROM deduped_vitals
                  GROUP BY 1, 2
                  HAVING COUNT(*) >= 5 -- Still require 5 samples for statistical significance
                )
                SELECT
                  UNIX_MILLIS(s.hour_bucket) AS timestamp,
                  s.metric_name,
                  s.app_version,
                  s.p50_val,
                  s.p75_val AS average_value,
                  s.p90_val
                FROM hourly_stats s
                WHERE s.hour_bucket < TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), HOUR)
                ORDER BY 1 ASC
            """

            query_job = bq_client.query(query)
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
            date_ranges=[DateRange(start_date=start_date_str, end_date=end_date_str)],
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

        # NOTE: We do not cache the GA4 fallback data because it uses simple
        # averages (mathematically incorrect for delta-based metrics) and
        # lacks percentiles. It is strictly a temporary fallback.
        return jsonify(performance_data)

    except Exception as e:
        current_app.logger.error(f"Error fetching performance analytics data: {e}")
        return jsonify({"error": f"Failed to fetch performance data: {e}"}), 500


@analytics_bp.route("/api/events", methods=["POST"])
def track_event() -> tuple[Response, int] | Response:
    """Receive and relay client analytics events to GA4.

    Returns:
        tuple[Response, int] | Response: JSON status payload and HTTP status code.
    """
    try:
        data = request.get_json(silent=True)
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
