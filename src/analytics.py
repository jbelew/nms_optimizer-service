"""
GA4 server-side event tracking via Google Analytics Measurement Protocol.

This module handles sending analytics events to GA4 from the backend when
client-side tracking is blocked. It uses the Measurement Protocol API to
relay validated events with client session, User-Agent, IP geo override,
and user property context.

Reference: https://developers.google.com/analytics/devguides/collection/protocol/ga4
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import math
import os
import re
import threading
from typing import Any, Dict, Optional
import uuid
import requests

logger = logging.getLogger(__name__)

# Suppress verbose logging from requests and urllib3 to avoid leaking secrets
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# GA4 Configuration
GA4_MEASUREMENT_ID = os.environ.get("GA4_MEASUREMENT_ID")
GA4_API_SECRET = os.environ.get("GA4_API_SECRET")
GA4_MEASUREMENT_PROTOCOL_URL = "https://www.google-analytics.com/mp/collect"

# Valid GA4 event name and parameter key patterns (alphanumeric and underscore)
VALID_EVENT_NAME_PATTERN = r"^[a-zA-Z0-9_]+$"
VALID_PARAM_KEY_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

# Payload and schema bounds matching GA4 Measurement Protocol limits
MAX_CLIENT_ID_LENGTH = 128
MAX_EVENT_PARAMS = 25
MAX_PARAM_KEY_LENGTH = 40
MAX_PARAM_STR_LENGTH = 500
MAX_USER_PROPERTIES = 25
MAX_USER_PROPERTY_KEY_LENGTH = 24
MAX_USER_PROPERTY_STR_LENGTH = 36
MAX_USER_AGENT_LENGTH = 512
MAX_IN_FLIGHT_EVENTS = 64

# Explicit allowlist of client-relayed GA4Event actions accepted by POST /api/events
ALLOWED_CLIENT_EVENT_NAMES: frozenset[str] = frozenset(
    {
        "app_installed",
        "apply_build",
        "buy_me_a_coffee_click",
        "earn_virtual_currency",
        "exception",
        "load_build",
        "no_fit_warning",
        "not_found",
        "optimize_tech",
        "page_view",
        "performance_metric",
        "reset_grid",
        "save_build",
        "screenshot_build",
        "select_content",
        "share",
        "slot_restriction_warning",
    }
)


def sanitize_event_params(params: Dict[str, Any]) -> Dict[str, str | int | float]:
    """Sanitize and bound event parameters before forwarding to GA4 Measurement Protocol.

    Normalizes Python booleans into ``"true"`` / ``"false"`` strings (preserving BigQuery
    ``string_value`` parity with client ``gtag``), retains finite ``int`` and ``float`` values
    (required for ``session_id`` and ``engagement_time_msec``), truncates strings, and caps
    parameter count to GA4's 25-parameter limit.

    Args:
        params: Raw client event parameter dictionary.

    Returns:
        Dict[str, str | int | float]: Sanitized parameter dictionary safe for GA4 ingest.
    """
    sanitized: Dict[str, str | int | float] = {}
    for key, value in params.items():
        if len(sanitized) >= MAX_EVENT_PARAMS:
            break
        if not isinstance(key, str) or not key or len(key) > MAX_PARAM_KEY_LENGTH:
            continue
        if not VALID_PARAM_KEY_PATTERN.match(key):
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            sanitized[key] = "true" if value else "false"
        elif isinstance(value, int):
            sanitized[key] = value
        elif isinstance(value, float):
            if math.isfinite(value):
                sanitized[key] = value
        elif isinstance(value, str):
            sanitized[key] = value[:MAX_PARAM_STR_LENGTH]
    return sanitized


def format_mp_user_properties(
    user_properties: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, str | int | float]]:
    """Format user properties into GA4 Measurement Protocol ``{"value": ...}`` objects.

    Args:
        user_properties: Optional mapping of user property keys to scalar values.

    Returns:
        Dict[str, Dict[str, str | int | float]]: Measurement Protocol ``user_properties`` map.
    """
    if not user_properties or not isinstance(user_properties, dict):
        return {}

    formatted: Dict[str, Dict[str, str | int | float]] = {}
    for key, value in user_properties.items():
        if len(formatted) >= MAX_USER_PROPERTIES:
            break
        if not isinstance(key, str) or not key or len(key) > MAX_USER_PROPERTY_KEY_LENGTH:
            continue
        if not VALID_PARAM_KEY_PATTERN.match(key):
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            formatted[key] = {"value": "true" if value else "false"}
        elif isinstance(value, int):
            formatted[key] = {"value": value}
        elif isinstance(value, float):
            if math.isfinite(value):
                formatted[key] = {"value": value}
        elif isinstance(value, str):
            formatted[key] = {"value": value[:MAX_USER_PROPERTY_STR_LENGTH]}
    return formatted


@dataclass
class AnalyticsEvent:
    """Represents a GA4 event to be sent to the Measurement Protocol."""

    name: str  # GA4 event name (e.g., "optimize_tech")
    params: Dict[str, Any]  # Event parameters
    client_id: Optional[str] = None  # Client ID for user tracking
    user_id: Optional[str] = None  # Optional user ID
    timestamp_micros: Optional[int] = None  # Event timestamp in microseconds
    user_properties: Optional[Dict[str, Any]] = None  # User-scoped properties
    user_agent: Optional[str] = None  # End-user browser User-Agent string
    ip_override: Optional[str] = None  # End-user IP address for coarse GA4 geo resolution

    def to_measurement_protocol_dict(self) -> Dict[str, Any]:
        """Convert event to Measurement Protocol event-item format without mutating ``self``.

        Returns:
            Dict[str, Any]: Measurement Protocol event dictionary containing ``name``,
            ``params``, and optional ``timestamp_micros``.
        """
        event_dict: Dict[str, Any] = {
            "name": self.name,
            "params": self.params,
        }

        if self.timestamp_micros:
            event_dict["timestamp_micros"] = self.timestamp_micros

        return event_dict


class GA4Client:
    """Client for sending events to GA4 via the Measurement Protocol."""

    def __init__(self, measurement_id: Optional[str] = None, api_secret: Optional[str] = None) -> None:
        """Initialize GA4 client.

        Args:
            measurement_id: GA4 Measurement ID (usually starts with ``G-``).
            api_secret: GA4 API Secret for Measurement Protocol.
        """
        self.measurement_id = measurement_id or os.environ.get("GA4_MEASUREMENT_ID")
        self.api_secret = api_secret or os.environ.get("GA4_API_SECRET")
        self.enabled = bool(self.measurement_id and self.api_secret)

        if not self.enabled:
            logger.debug("GA4 not fully configured (missing measurement_id or api_secret). Analytics disabled.")

    def send_event(self, event: AnalyticsEvent) -> bool:
        """Send a single event to GA4.

        Args:
            event: ``AnalyticsEvent`` to send.

        Returns:
            bool: ``True`` if accepted by GA4 (HTTP 204), ``False`` otherwise.
        """
        if not self.enabled:
            return False

        if not self._validate_event(event):
            return False

        try:
            client_id = event.client_id or str(uuid.uuid4())
            payload: Dict[str, Any] = {
                "client_id": client_id,
                "events": [event.to_measurement_protocol_dict()],
            }

            if event.user_id:
                payload["user_id"] = event.user_id

            mp_user_props = format_mp_user_properties(event.user_properties)
            if mp_user_props:
                payload["user_properties"] = mp_user_props

            if event.user_agent:
                payload["user_agent"] = event.user_agent[:MAX_USER_AGENT_LENGTH]

            if event.ip_override:
                payload["ip_override"] = event.ip_override

            logger.debug(f"Sending GA4 event: {event.name} (event_count=1)")

            url = f"{GA4_MEASUREMENT_PROTOCOL_URL}?measurement_id={self.measurement_id}&api_secret={self.api_secret}"

            response = requests.post(
                url,
                json=payload,
                timeout=5,
            )

            logger.debug(f"GA4 response status: {response.status_code}")

            if response.status_code == 204:
                logger.info(f"GA4 event sent: {event.name}")
                return True

            logger.warning(f"GA4 event failed (status {response.status_code})")
            return False

        except requests.exceptions.Timeout:
            logger.error(f"GA4 request timeout sending event: {event.name}")
            return False
        except requests.exceptions.ConnectionError as e:
            logger.error(f"GA4 connection error: {e}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"GA4 request error: {e}")
            return False
        except ValueError as e:
            logger.error(f"Invalid event data: {e}")
            return False

    def send_events(self, events: list[AnalyticsEvent]) -> bool:
        """Send multiple events in batch to GA4.

        Args:
            events: List of ``AnalyticsEvent`` objects.

        Returns:
            bool: ``True`` if accepted by GA4 (HTTP 204), ``False`` otherwise.
        """
        if not self.enabled or not events:
            return False

        for event in events:
            if not self._validate_event(event):
                return False

        try:
            first_event = events[0]
            default_client_id = first_event.client_id or str(uuid.uuid4())

            payload: Dict[str, Any] = {
                "client_id": default_client_id,
                "events": [event.to_measurement_protocol_dict() for event in events],
            }

            if first_event.user_id:
                payload["user_id"] = first_event.user_id

            mp_user_props = format_mp_user_properties(first_event.user_properties)
            if mp_user_props:
                payload["user_properties"] = mp_user_props

            if first_event.user_agent:
                payload["user_agent"] = first_event.user_agent[:MAX_USER_AGENT_LENGTH]

            if first_event.ip_override:
                payload["ip_override"] = first_event.ip_override

            logger.debug(f"Sending GA4 batch: {len(events)} events")

            url = f"{GA4_MEASUREMENT_PROTOCOL_URL}?measurement_id={self.measurement_id}&api_secret={self.api_secret}"

            response = requests.post(
                url,
                json=payload,
                timeout=5,
            )

            logger.debug(f"GA4 batch response status: {response.status_code}")

            if response.status_code == 204:
                logger.info(f"GA4 batch sent: {len(events)} events")
                return True

            logger.warning(f"GA4 batch failed (status {response.status_code})")
            return False

        except requests.exceptions.Timeout:
            logger.error(f"GA4 batch request timeout: {len(events)} events")
            return False
        except requests.exceptions.ConnectionError as e:
            logger.error(f"GA4 connection error: {e}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"GA4 batch request error: {e}")
            return False
        except ValueError as e:
            logger.error(f"Invalid event data in batch: {e}")
            return False

    def _validate_event(self, event: AnalyticsEvent) -> bool:
        """Validate event structure before sending.

        Args:
            event: Event to validate.

        Returns:
            bool: ``True`` if valid, ``False`` otherwise.
        """
        if not event.name or not isinstance(event.name, str):
            logger.error("Event name is required and must be a string")
            return False

        if not re.match(VALID_EVENT_NAME_PATTERN, event.name):
            logger.error(f"Invalid event name format: {event.name}")
            return False

        if not isinstance(event.params, dict):
            logger.error("Event params must be a dictionary")
            return False

        return True


# Singleton instance
_ga4_client = GA4Client()

# Fork-safe worker-local background executor for non-blocking /api/events dispatch
_executor_lock = threading.Lock()
_executor_pid: Optional[int] = None
_executor: Optional[ThreadPoolExecutor] = None
_in_flight_semaphore = threading.BoundedSemaphore(MAX_IN_FLIGHT_EVENTS)


def _get_worker_executor() -> ThreadPoolExecutor:
    """Return a PID-scoped ``ThreadPoolExecutor`` safe across Gunicorn ``--preload`` forks.

    Returns:
        ThreadPoolExecutor: Active background executor for the current worker process.
    """
    global _executor, _executor_pid, _in_flight_semaphore
    current_pid = os.getpid()
    if _executor is None or _executor_pid != current_pid:
        with _executor_lock:
            if _executor is None or _executor_pid != current_pid:
                _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ga4-mp")
                _executor_pid = current_pid
                _in_flight_semaphore = threading.BoundedSemaphore(MAX_IN_FLIGHT_EVENTS)
    return _executor


def send_analytics_event(
    event_name: str,
    client_id: str,
    params: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    user_properties: Optional[Dict[str, Any]] = None,
    user_agent: Optional[str] = None,
    ip_override: Optional[str] = None,
) -> bool:
    """Send an analytics event synchronously to GA4.

    Args:
        event_name: Name of the event (e.g., ``"optimize_tech"``).
        client_id: Unique client identifier.
        params: Event parameters dict.
        user_id: Optional user ID.
        user_properties: Optional user-scoped properties dict.
        user_agent: Optional client ``User-Agent`` string.
        ip_override: Optional client IP address for coarse GA4 geo resolution.

    Returns:
        bool: ``True`` if event was sent successfully, ``False`` otherwise.
    """
    event = AnalyticsEvent(
        name=event_name,
        params={} if params is None else params,
        client_id=client_id,
        user_id=user_id,
        user_properties=user_properties,
        user_agent=user_agent,
        ip_override=ip_override,
    )

    return _ga4_client.send_event(event)


def enqueue_analytics_event(
    event_name: str,
    client_id: str,
    params: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    user_properties: Optional[Dict[str, Any]] = None,
    user_agent: Optional[str] = None,
    ip_override: Optional[str] = None,
) -> bool:
    """Validate and enqueue a client-relayed analytics event for non-blocking GA4 delivery.

    Args:
        event_name: Allowlisted GA4 event name.
        client_id: Unique client identifier.
        params: Sanitized event parameters dict.
        user_id: Optional user ID.
        user_properties: Optional user-scoped properties dict.
        user_agent: Optional client ``User-Agent`` string.
        ip_override: Optional client IP address for coarse GA4 geo resolution.

    Returns:
        bool: ``True`` when accepted for delivery (or no-op when GA4 credentials are unset),
        ``False`` when validation fails or the in-flight queue is saturated.
    """
    event = AnalyticsEvent(
        name=event_name,
        params={} if params is None else params,
        client_id=client_id,
        user_id=user_id,
        user_properties=user_properties,
        user_agent=user_agent,
        ip_override=ip_override,
    )

    if not _ga4_client._validate_event(event):
        return False

    if not _ga4_client.enabled:
        return True

    if not _in_flight_semaphore.acquire(blocking=False):
        logger.warning("Dropping GA4 relay event due to saturated in-flight queue")
        return False

    def _deliver() -> None:
        try:
            _ga4_client.send_event(event)
        finally:
            _in_flight_semaphore.release()

    try:
        _get_worker_executor().submit(_deliver)
        return True
    except RuntimeError:
        _in_flight_semaphore.release()
        return False


def send_analytics_batch(events: list[AnalyticsEvent]) -> bool:
    """Send multiple analytics events in a batch to GA4.

    Args:
        events: List of ``AnalyticsEvent`` objects.

    Returns:
        bool: ``True`` if batch was sent successfully, ``False`` otherwise.
    """
    return _ga4_client.send_events(events)
