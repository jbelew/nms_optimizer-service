"""
GA4 server-side event tracking via Google Analytics Measurement Protocol.

This module handles sending analytics events to GA4 from the backend,
replacing client-side tracking. It uses the Measurement Protocol API
to send events with server-side context.

Reference: https://developers.google.com/analytics/devguides/collection/protocol/ga4
"""

import os
import re
import logging
import uuid
from typing import Any, Dict, Optional
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)

# Suppress verbose logging from requests and urllib3 to avoid leaking secrets
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# GA4 Configuration
GA4_MEASUREMENT_ID = os.environ.get("GA4_MEASUREMENT_ID")
GA4_API_SECRET = os.environ.get("GA4_API_SECRET")
GA4_MEASUREMENT_PROTOCOL_URL = "https://www.google-analytics.com/mp/collect"

# Valid GA4 event name pattern (alphanumeric and underscore)
VALID_EVENT_NAME_PATTERN = r"^[a-zA-Z0-9_]+$"


@dataclass
class AnalyticsEvent:
    """Represents a GA4 event to be sent to the Measurement Protocol."""

    name: str  # GA4 event name (e.g., "optimization_complete")
    params: Dict[str, Any]  # Event parameters
    client_id: Optional[str] = None  # Client ID for user tracking
    user_id: Optional[str] = None  # Optional user ID
    timestamp_micros: Optional[int] = None  # Event timestamp in microseconds

    def to_measurement_protocol_dict(self) -> Dict[str, Any]:
        """Convert event to Measurement Protocol format.

        Note: Does not mutate self. Generates temp client_id if not set.
        """
        event_dict = {
            "name": self.name,
            "params": self.params,
        }

        if self.timestamp_micros:
            event_dict["timestamp_micros"] = self.timestamp_micros

        return event_dict


class GA4Client:
    """Client for sending events to GA4 via the Measurement Protocol."""

    def __init__(self, measurement_id: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize GA4 client.

        Args:
                measurement_id: GA4 Measurement ID (usually starts with G-)
                api_secret: GA4 API Secret for Measurement Protocol
        """
        self.measurement_id = measurement_id or GA4_MEASUREMENT_ID
        self.api_secret = api_secret or GA4_API_SECRET
        self.enabled = bool(self.measurement_id and self.api_secret)

        if not self.enabled:
            logger.debug("GA4 not fully configured (missing measurement_id or api_secret). Analytics disabled.")

    def send_event(self, event: AnalyticsEvent) -> bool:
        """Send a single event to GA4.

        Args:
                event: AnalyticsEvent to send

        Returns:
                True if successful, False otherwise
        """
        if not self.enabled:
            return False

        if not self._validate_event(event):
            return False

        try:
            client_id = event.client_id or str(uuid.uuid4())
            payload = {
                "client_id": client_id,
                "events": [event.to_measurement_protocol_dict()],
            }

            if event.user_id:
                payload["user_id"] = event.user_id

            # Log only non-sensitive event info
            logger.debug(f"Sending GA4 event: {event.name} (event_count=1)")

            # Use query params as required by GA4 Measurement Protocol
            # (secrets in URLs are a limitation of this API)
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
            else:
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
                events: List of AnalyticsEvent objects

        Returns:
                True if successful, False otherwise
        """
        if not self.enabled or not events:
            return False

        # Validate all events
        for event in events:
            if not self._validate_event(event):
                return False

        try:
            # Use first event's client_id for all if not specified
            default_client_id = events[0].client_id or str(uuid.uuid4())

            payload = {
                "client_id": default_client_id,
                "events": [event.to_measurement_protocol_dict() for event in events],
            }

            # Log only non-sensitive info
            logger.debug(f"Sending GA4 batch: {len(events)} events")

            # Use query params as required by GA4 Measurement Protocol
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
            else:
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
        """Validate event before sending.

        Args:
                event: Event to validate

        Returns:
                True if valid, False otherwise
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


def send_analytics_event(
    event_name: str,
    client_id: str,
    params: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> bool:
    """Send an analytics event to GA4.

    Args:
            event_name: Name of the event (e.g., "optimization_complete")
            client_id: Unique client identifier
            params: Event parameters dict
            user_id: Optional user ID

    Returns:
            True if event was sent successfully, False otherwise
    """
    if params is None:
        params = {}

    event = AnalyticsEvent(
        name=event_name,
        params=params,
        client_id=client_id,
        user_id=user_id,
    )

    return _ga4_client.send_event(event)


def send_analytics_batch(events: list[AnalyticsEvent]) -> bool:
    """Send multiple analytics events in a batch to GA4.

    Args:
            events: List of AnalyticsEvent objects

    Returns:
            True if batch was sent successfully, False otherwise
    """
    return _ga4_client.send_events(events)
