"""Tests for GA4 analytics module."""

import unittest
from unittest.mock import patch, MagicMock
import uuid
from src.analytics import (
    AnalyticsEvent,
    GA4Client,
    send_analytics_event,
    send_analytics_batch,
    VALID_EVENT_NAME_PATTERN,
)


class TestAnalyticsEvent(unittest.TestCase):
    """Test AnalyticsEvent dataclass."""

    def test_event_creation_minimal(self):
        """Test creating event with minimal parameters."""
        event = AnalyticsEvent(name="test_event", params={})
        self.assertEqual(event.name, "test_event")
        self.assertEqual(event.params, {})
        self.assertIsNone(event.client_id)
        self.assertIsNone(event.user_id)
        self.assertIsNone(event.timestamp_micros)

    def test_event_creation_full(self):
        """Test creating event with all parameters."""
        event = AnalyticsEvent(
            name="test_event",
            params={"key": "value"},
            client_id="client-123",
            user_id="user-456",
            timestamp_micros=1234567890,
        )
        self.assertEqual(event.name, "test_event")
        self.assertEqual(event.params, {"key": "value"})
        self.assertEqual(event.client_id, "client-123")
        self.assertEqual(event.user_id, "user-456")
        self.assertEqual(event.timestamp_micros, 1234567890)

    def test_to_measurement_protocol_dict_basic(self):
        """Test conversion to measurement protocol format."""
        event = AnalyticsEvent(name="test_event", params={"key": "value"})
        result = event.to_measurement_protocol_dict()

        self.assertEqual(result["name"], "test_event")
        self.assertEqual(result["params"], {"key": "value"})
        self.assertNotIn("timestamp_micros", result)

    def test_to_measurement_protocol_dict_with_timestamp(self):
        """Test conversion includes timestamp when provided."""
        event = AnalyticsEvent(name="test_event", params={}, timestamp_micros=1234567890)
        result = event.to_measurement_protocol_dict()

        self.assertEqual(result["timestamp_micros"], 1234567890)

    def test_to_measurement_protocol_dict_does_not_mutate(self):
        """Test that conversion doesn't mutate the original event."""
        event = AnalyticsEvent(name="test_event", params={})
        original_client_id = event.client_id

        event.to_measurement_protocol_dict()

        self.assertEqual(event.client_id, original_client_id)


class TestGA4Client(unittest.TestCase):
    """Test GA4Client class."""

    def test_client_init_with_env_vars(self):
        """Test client initialization with environment variables."""
        with patch("src.analytics.GA4_MEASUREMENT_ID", "G-TEST"):
            with patch("src.analytics.GA4_API_SECRET", "secret-123"):
                client = GA4Client()
                self.assertEqual(client.measurement_id, "G-TEST")
                self.assertEqual(client.api_secret, "secret-123")
                self.assertTrue(client.enabled)

    def test_client_init_disabled_no_secret(self):
        """Test client is disabled when api_secret is missing."""
        with patch.dict("os.environ", {"GA4_MEASUREMENT_ID": "G-TEST"}, clear=True):
            client = GA4Client()
            self.assertFalse(client.enabled)

    def test_client_init_disabled_no_measurement_id(self):
        """Test client is disabled when measurement_id is missing."""
        with patch.dict("os.environ", {"GA4_API_SECRET": "secret-123"}, clear=True):
            client = GA4Client()
            self.assertFalse(client.enabled)

    def test_client_init_override_params(self):
        """Test client initialization with explicit parameters."""
        client = GA4Client(measurement_id="G-OVERRIDE", api_secret="secret-override")
        self.assertEqual(client.measurement_id, "G-OVERRIDE")
        self.assertEqual(client.api_secret, "secret-override")
        self.assertTrue(client.enabled)

    @patch("requests.post")
    def test_send_event_success(self, mock_post):
        """Test successful event send."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        event = AnalyticsEvent(name="test_event", params={}, client_id="client-123")

        result = client.send_event(event)

        self.assertTrue(result)
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_send_event_disabled(self, mock_post):
        """Test send_event returns False when disabled."""
        client = GA4Client()  # No credentials, so disabled
        event = AnalyticsEvent(name="test_event", params={})

        result = client.send_event(event)

        self.assertFalse(result)
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_send_event_invalid_name(self, mock_post):
        """Test send_event returns False for invalid event name."""
        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        event = AnalyticsEvent(name="invalid-event-name", params={})

        result = client.send_event(event)

        self.assertFalse(result)
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_send_event_invalid_params(self, mock_post):
        """Test send_event returns False when params is not a dict."""
        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        event = AnalyticsEvent(name="test_event", params="invalid")  # type: ignore

        result = client.send_event(event)

        self.assertFalse(result)
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_send_event_with_user_id(self, mock_post):
        """Test send_event includes user_id in payload."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        event = AnalyticsEvent(name="test_event", params={}, client_id="client-123", user_id="user-456")

        result = client.send_event(event)

        self.assertTrue(result)
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        self.assertEqual(payload["user_id"], "user-456")

    @patch("requests.post")
    def test_send_event_generates_client_id(self, mock_post):
        """Test send_event generates client_id if not provided."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        event = AnalyticsEvent(name="test_event", params={})

        result = client.send_event(event)

        self.assertTrue(result)
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        # Should have generated a UUID
        self.assertIsNotNone(payload["client_id"])
        try:
            uuid.UUID(payload["client_id"])
        except ValueError:
            self.fail("Generated client_id is not a valid UUID")

    @patch("requests.post")
    def test_send_event_http_error(self, mock_post):
        """Test send_event returns False on HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        event = AnalyticsEvent(name="test_event", params={}, client_id="client-123")

        result = client.send_event(event)

        self.assertFalse(result)

    @patch("requests.post")
    def test_send_event_timeout(self, mock_post):
        """Test send_event handles timeout exception."""
        import requests

        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        event = AnalyticsEvent(name="test_event", params={}, client_id="client-123")

        result = client.send_event(event)

        self.assertFalse(result)

    @patch("requests.post")
    def test_send_events_batch_success(self, mock_post):
        """Test successful batch event send."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        events = [
            AnalyticsEvent(name="event1", params={"key": "value1"}),
            AnalyticsEvent(name="event2", params={"key": "value2"}),
        ]

        result = client.send_events(events)

        self.assertTrue(result)
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        self.assertEqual(len(payload["events"]), 2)

    @patch("requests.post")
    def test_send_events_empty_list(self, mock_post):
        """Test send_events returns False for empty list."""
        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")

        result = client.send_events([])

        self.assertFalse(result)
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_send_events_validation_failure(self, mock_post):
        """Test send_events returns False when any event is invalid."""
        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        events = [
            AnalyticsEvent(name="valid_event", params={}),
            AnalyticsEvent(name="invalid-event", params={}),  # Invalid name
        ]

        result = client.send_events(events)

        self.assertFalse(result)
        mock_post.assert_not_called()

    def test_validate_event_valid(self):
        """Test _validate_event with valid event."""
        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        event = AnalyticsEvent(name="valid_event_123", params={"key": "value"})

        result = client._validate_event(event)

        self.assertTrue(result)

    def test_validate_event_invalid_name_format(self):
        """Test _validate_event rejects invalid event name format."""
        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        event = AnalyticsEvent(name="invalid-event-name", params={})

        result = client._validate_event(event)

        self.assertFalse(result)

    def test_validate_event_empty_name(self):
        """Test _validate_event rejects empty event name."""
        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        event = AnalyticsEvent(name="", params={})

        result = client._validate_event(event)

        self.assertFalse(result)

    def test_validate_event_non_string_name(self):
        """Test _validate_event rejects non-string event name."""
        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        event = AnalyticsEvent(name=123, params={})  # type: ignore

        result = client._validate_event(event)

        self.assertFalse(result)

    def test_validate_event_non_dict_params(self):
        """Test _validate_event rejects non-dict params."""
        client = GA4Client(measurement_id="G-TEST", api_secret="secret-123")
        event = AnalyticsEvent(name="test_event", params="invalid")  # type: ignore

        result = client._validate_event(event)

        self.assertFalse(result)


class TestPublicFunctions(unittest.TestCase):
    """Test public module functions."""

    @patch("src.analytics._ga4_client.send_event")
    def test_send_analytics_event(self, mock_send):
        """Test send_analytics_event function."""
        mock_send.return_value = True

        result = send_analytics_event("test_event", "client-123", {"key": "value"})

        self.assertTrue(result)
        mock_send.assert_called_once()

    @patch("src.analytics._ga4_client.send_event")
    def test_send_analytics_event_default_params(self, mock_send):
        """Test send_analytics_event with default params."""
        mock_send.return_value = True

        result = send_analytics_event("test_event", "client-123")

        self.assertTrue(result)
        call_args = mock_send.call_args
        event = call_args[0][0]
        self.assertEqual(event.params, {})

    @patch("src.analytics._ga4_client.send_events")
    def test_send_analytics_batch(self, mock_send):
        """Test send_analytics_batch function."""
        mock_send.return_value = True

        events = [
            AnalyticsEvent(name="event1", params={}),
            AnalyticsEvent(name="event2", params={}),
        ]

        result = send_analytics_batch(events)

        self.assertTrue(result)
        mock_send.assert_called_once_with(events)


class TestEventNameValidation(unittest.TestCase):
    """Test event name pattern validation."""

    def test_valid_event_names(self):
        """Test that valid event names match pattern."""
        valid_names = [
            "optimization_complete",
            "event",
            "EVENT",
            "event123",
            "event_123",
            "e",
            "a_b_c_123",
        ]

        import re

        for name in valid_names:
            with self.subTest(name=name):
                self.assertIsNotNone(
                    re.match(VALID_EVENT_NAME_PATTERN, name),
                    f"Expected '{name}' to match pattern",
                )

    def test_invalid_event_names(self):
        """Test that invalid event names don't match pattern."""
        invalid_names = [
            "event-name",
            "event.name",
            "event name",
            "event@name",
            "event#name",
            "",
        ]

        import re

        for name in invalid_names:
            with self.subTest(name=name):
                self.assertIsNone(
                    re.match(VALID_EVENT_NAME_PATTERN, name),
                    f"Expected '{name}' to NOT match pattern",
                )


if __name__ == "__main__":
    unittest.main()
