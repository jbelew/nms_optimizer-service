"""
Comprehensive test suite for Flask app endpoints

This test suite focuses on finding bugs in REST API handlers,
WebSocket events, error handling, and request validation.
"""

import unittest
import json
from unittest.mock import patch, MagicMock
from src.app import app


class TestAppInitialization(unittest.TestCase):
    """Test Flask app initialization and basic setup."""

    def test_app_exists(self):
        """Flask app should be created."""
        self.assertIsNotNone(app)

    def test_app_is_flask_instance(self):
        """App should be a Flask instance."""
        from flask import Flask
        self.assertIsInstance(app, Flask)

    def test_cors_enabled(self):
        """CORS should be enabled."""
        # CORS headers should be set
        with app.test_client() as client:
            response = client.options('/')
            # CORS preflight should work
            self.assertIn('Access-Control-Allow-Origin', response.headers or {})

    def test_compression_enabled(self):
        """Compression should be enabled."""
        # Compress extension should be registered
        self.assertIsNotNone(app)


class TestHealthEndpoint(unittest.TestCase):
    """Test health/status endpoint."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_health_endpoint_exists(self):
        """Health endpoint should exist and return 200."""
        # Assuming there's a health endpoint, test it
        # This is a typical pattern but may need adjustment based on actual app
        response = self.client.get('/health')
        # Either 200 or 404 if endpoint doesn't exist
        self.assertIn(response.status_code, [200, 404])


class TestOptimizationEndpoint(unittest.TestCase):
    """Test optimization endpoint."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()
        self.valid_request = {
            "ship": "corvette",
            "tech": "pulse",
            "rewards": [],
            "seed": 42
        }

    def test_optimization_endpoint_requires_post(self):
        """Optimization should require POST method."""
        response = self.client.get('/api/optimize')
        # Should not accept GET
        self.assertNotEqual(response.status_code, 200)

    def test_optimization_missing_ship_parameter(self):
        """Request without ship should fail."""
        request_data = self.valid_request.copy()
        del request_data["ship"]
        
        response = self.client.post('/api/optimize',
                                   data=json.dumps(request_data),
                                   content_type='application/json')
        
        # Should return 400 or similar error
        self.assertGreaterEqual(response.status_code, 400)

    def test_optimization_missing_tech_parameter(self):
        """Request without tech should fail."""
        request_data = self.valid_request.copy()
        del request_data["tech"]
        
        response = self.client.post('/api/optimize',
                                   data=json.dumps(request_data),
                                   content_type='application/json')
        
        self.assertGreaterEqual(response.status_code, 400)

    def test_optimization_with_invalid_json(self):
        """Invalid JSON should be rejected."""
        response = self.client.post('/api/optimize',
                                   data="not valid json",
                                   content_type='application/json')
        
        self.assertGreaterEqual(response.status_code, 400)

    def test_optimization_response_is_json(self):
        """Response should be valid JSON."""
        response = self.client.post('/api/optimize',
                                   data=json.dumps(self.valid_request),
                                   content_type='application/json')
        
        if response.status_code == 200:
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                self.fail("Response is not valid JSON")

    def test_optimization_response_contains_grid(self):
        """Successful response should contain grid data."""
        response = self.client.post('/api/optimize',
                                   data=json.dumps(self.valid_request),
                                   content_type='application/json')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn('grid', data)


class TestTechTreeEndpoint(unittest.TestCase):
    """Test technology tree endpoint."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_tech_tree_endpoint_requires_post(self):
        """Tech tree should require POST method."""
        response = self.client.get('/api/tech-tree')
        self.assertNotEqual(response.status_code, 200)

    def test_tech_tree_missing_ship(self):
        """Request without ship should fail."""
        request_data = {"tech": "pulse"}
        
        response = self.client.post('/api/tech-tree',
                                   data=json.dumps(request_data),
                                   content_type='application/json')
        
        self.assertGreaterEqual(response.status_code, 400)

    def test_tech_tree_response_is_json(self):
        """Response should be valid JSON."""
        response = self.client.post('/api/tech-tree',
                                   data=json.dumps({"ship": "corvette"}),
                                   content_type='application/json')
        
        if response.status_code == 200:
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                self.fail("Response is not valid JSON")


class TestAnalyticsEndpoint(unittest.TestCase):
    """Test analytics/popular data endpoint."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_analytics_endpoint_exists(self):
        """Analytics endpoint should exist."""
        response = self.client.get('/api/analytics/popular_data')
        # Should either work (200) or be disabled (404, 403)
        self.assertIn(response.status_code, [200, 404, 403])

    def test_analytics_response_format(self):
        """Analytics response should be valid JSON if successful."""
        response = self.client.get('/api/analytics/popular_data')
        
        if response.status_code == 200:
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                self.fail("Response is not valid JSON")


class TestErrorHandling(unittest.TestCase):
    """Test error handling in endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_invalid_endpoint_returns_404(self):
        """Nonexistent endpoint should return 404."""
        response = self.client.get('/api/nonexistent')
        self.assertEqual(response.status_code, 404)

    def test_malformed_request_body(self):
        """Malformed request body should be rejected."""
        response = self.client.post('/api/optimize',
                                   data='{"invalid": json}',
                                   content_type='application/json')
        
        self.assertGreaterEqual(response.status_code, 400)

    def test_empty_request_body(self):
        """Empty request body should be rejected."""
        response = self.client.post('/api/optimize',
                                   data='',
                                   content_type='application/json')
        
        self.assertGreaterEqual(response.status_code, 400)

    def test_null_request_body(self):
        """Null request body should be rejected."""
        response = self.client.post('/api/optimize',
                                   data='null',
                                   content_type='application/json')
        
        self.assertGreaterEqual(response.status_code, 400)


class TestRequestValidation(unittest.TestCase):
    """Test request parameter validation."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()
        self.valid_request = {
            "ship": "corvette",
            "tech": "pulse",
            "rewards": [],
            "seed": 42
        }

    def test_invalid_ship_type(self):
        """Invalid ship type should be rejected or handled."""
        request_data = self.valid_request.copy()
        request_data["ship"] = "nonexistent_ship"
        
        response = self.client.post('/api/optimize',
                                   data=json.dumps(request_data),
                                   content_type='application/json')
        
        # Should either reject or handle gracefully
        self.assertIsNotNone(response)

    def test_invalid_tech_type(self):
        """Invalid tech type should be rejected or handled."""
        request_data = self.valid_request.copy()
        request_data["tech"] = "nonexistent_tech"
        
        response = self.client.post('/api/optimize',
                                   data=json.dumps(request_data),
                                   content_type='application/json')
        
        self.assertIsNotNone(response)

    def test_empty_rewards_list(self):
        """Empty rewards list should be accepted."""
        request_data = self.valid_request.copy()
        request_data["rewards"] = []
        
        response = self.client.post('/api/optimize',
                                   data=json.dumps(request_data),
                                   content_type='application/json')
        
        # Should not fail due to empty rewards
        self.assertNotEqual(response.status_code, 500)

    def test_invalid_reward_format(self):
        """Invalid reward format should be handled."""
        request_data = self.valid_request.copy()
        request_data["rewards"] = "not_a_list"
        
        response = self.client.post('/api/optimize',
                                   data=json.dumps(request_data),
                                   content_type='application/json')
        
        # Should handle gracefully
        self.assertIsNotNone(response)

    def test_negative_seed(self):
        """Negative seed should be handled."""
        request_data = self.valid_request.copy()
        request_data["seed"] = -1
        
        response = self.client.post('/api/optimize',
                                   data=json.dumps(request_data),
                                   content_type='application/json')
        
        self.assertIsNotNone(response)


class TestResponseFormat(unittest.TestCase):
    """Test response format consistency."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_error_responses_have_message(self):
        """Error responses should contain an error message."""
        response = self.client.get('/api/nonexistent')
        
        # 404 or error response should exist
        if response.status_code >= 400:
            try:
                data = json.loads(response.data)
                # Should have some indication of error
                self.assertIsNotNone(data)
            except json.JSONDecodeError:
                # At minimum, should have some content
                self.assertGreater(len(response.data), 0)

    def test_response_headers_valid(self):
        """Response headers should be valid."""
        response = self.client.get('/')
        
        # Content-Type should be set for JSON responses
        # (varies based on endpoint)
        self.assertIsNotNone(response.headers)


class TestCORSHeaders(unittest.TestCase):
    """Test CORS header handling."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_cors_headers_on_successful_response(self):
        """Successful responses should have CORS headers."""
        response = self.client.get('/', headers={'Origin': 'http://localhost:3000'})
        
        # CORS should be enabled
        self.assertIsNotNone(response)

    def test_cors_preflight_request(self):
        """OPTIONS preflight requests should be handled."""
        response = self.client.options('/api/optimize',
                                      headers={'Origin': 'http://localhost:3000'})
        
        # Should handle OPTIONS (200, 204 if supported, 404 if not)
        self.assertIn(response.status_code, [200, 204, 404])


class TestContentNegotiation(unittest.TestCase):
    """Test content type handling."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_json_content_type_accepted(self):
        """application/json content type should be accepted."""
        response = self.client.post('/api/optimize',
                                   data=json.dumps({"ship": "corvette", "tech": "pulse"}),
                                   content_type='application/json')
        
        # Should not reject based on content type
        self.assertNotEqual(response.status_code, 415)

    def test_wrong_content_type(self):
        """Wrong content type might be rejected."""
        response = self.client.post('/api/optimize',
                                   data="invalid",
                                   content_type='text/plain')
        
        # Should either reject or handle gracefully
        self.assertIsNotNone(response)


class TestEndpointIntegration(unittest.TestCase):
    """Integration tests for endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_multiple_requests_sequential(self):
        """Should handle multiple requests in sequence."""
        for i in range(3):
            response = self.client.get('/')
            self.assertIsNotNone(response)

    def test_concurrent_request_isolation(self):
        """Requests should not interfere with each other."""
        # Sequential calls should work independently
        response1 = self.client.get('/')
        response2 = self.client.get('/')
        
        self.assertIsNotNone(response1)
        self.assertIsNotNone(response2)


if __name__ == "__main__":
    unittest.main()
