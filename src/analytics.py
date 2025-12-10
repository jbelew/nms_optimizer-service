"""
GA4 server-side event tracking via Google Analytics Measurement Protocol.

This module handles sending analytics events to GA4 from the backend,
replacing client-side tracking. It uses the Measurement Protocol API
to send events with server-side context.

Reference: https://developers.google.com/analytics/devguides/collection/protocol/ga4
"""

import os
import json
import logging
import uuid
from typing import Any, Dict, Optional
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)

# GA4 Configuration
GA4_MEASUREMENT_ID = os.environ.get("GA4_MEASUREMENT_ID", "G-P5VBZQ69Q9")
GA4_API_SECRET = os.environ.get("GA4_API_SECRET", "")
GA4_MEASUREMENT_PROTOCOL_URL = "https://www.google-analytics.com/mp/collect"


@dataclass
class AnalyticsEvent:
    """Represents a GA4 event to be sent to the Measurement Protocol."""

    name: str  # GA4 event name (e.g., "optimization_complete")
    params: Dict[str, Any]  # Event parameters
    client_id: Optional[str] = None  # Client ID for user tracking
    user_id: Optional[str] = None  # Optional user ID
    timestamp_micros: Optional[int] = None  # Event timestamp in microseconds

    def to_measurement_protocol_dict(self) -> Dict[str, Any]:
        """Convert event to Measurement Protocol format."""
        if not self.client_id:
            self.client_id = str(uuid.uuid4())

        event_dict = {
            "name": self.name,
            "params": self.params,
        }

        if self.timestamp_micros:
            event_dict["timestamp_micros"] = self.timestamp_micros

        return event_dict


class GA4Client:
    """Client for sending events to GA4 via the Measurement Protocol."""

    def __init__(self, measurement_id: str = GA4_MEASUREMENT_ID, api_secret: str = GA4_API_SECRET):
        """Initialize GA4 client.

        Args:
                measurement_id: GA4 Measurement ID (usually starts with G-)
                api_secret: GA4 API Secret for Measurement Protocol
        """
        self.measurement_id = measurement_id
        self.api_secret = api_secret
        self.enabled = bool(api_secret)  # Only send if API secret is configured

        if not self.enabled:
            logger.warning("GA4_API_SECRET not configured. Analytics events will not be sent.")

    def send_event(self, event: AnalyticsEvent) -> bool:
        """Send a single event to GA4.

        Args:
                event: AnalyticsEvent to send

        Returns:
                True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            payload = {
                "client_id": event.client_id or str(uuid.uuid4()),
                "events": [event.to_measurement_protocol_dict()],
            }

            if event.user_id:
                payload["user_id"] = event.user_id

            params = {
                "measurement_id": self.measurement_id,
                "api_secret": self.api_secret,
            }

            logger.debug(f"Sending GA4 event: {event.name} with payload: {json.dumps(payload, indent=2)}")

            response = requests.post(
                GA4_MEASUREMENT_PROTOCOL_URL,
                json=payload,
                params=params,
                timeout=5,
            )

            logger.debug(f"GA4 response status: {response.status_code}")
            if response.text:
                logger.debug(f"GA4 response body: {response.text}")

            if response.status_code == 204:
                logger.info(f"✓ GA4 event sent successfully: {event.name}")
                return True
            else:
                logger.warning(f"GA4 event failed (status {response.status_code}): {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending GA4 event: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in GA4 event tracking: {e}")
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

        try:
            # Use first event's client_id for all if not specified
            default_client_id = events[0].client_id or str(uuid.uuid4())

            payload = {
                "client_id": default_client_id,
                "events": [
                    {
                        **event.to_measurement_protocol_dict(),
                        # Ensure all events use same client_id for batching
                        "params": {
                            **event.params,
                        },
                    }
                    for event in events
                ],
            }

            params = {
                "measurement_id": self.measurement_id,
                "api_secret": self.api_secret,
            }

            logger.debug(f"Sending GA4 batch: {len(events)} events with payload: {json.dumps(payload, indent=2)}")

            response = requests.post(
                GA4_MEASUREMENT_PROTOCOL_URL,
                json=payload,
                params=params,
                timeout=5,
            )

            logger.debug(f"GA4 batch response status: {response.status_code}")

            if response.status_code == 204:
                logger.info(f"✓ GA4 batch sent successfully: {len(events)} events")
                return True
            else:
                logger.warning(f"GA4 batch failed (status {response.status_code}): {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending GA4 batch: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in GA4 batch tracking: {e}")
            return False


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
