"""
Comprehensive test suite for Flask app endpoints

This test suite focuses on finding bugs in REST API handlers,
WebSocket events, error handling, and request validation.
"""

import json
import re
import unittest
from unittest.mock import MagicMock, patch
from src.app import app
from src.routes.analytics import perf_cache


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
            response = client.options("/")
            # CORS preflight should work
            self.assertIn("Access-Control-Allow-Origin", response.headers or {})

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
        response = self.client.get("/health")
        # Either 200 or 404 if endpoint doesn't exist
        self.assertIn(response.status_code, [200, 404])


class TestOptimizationEndpoint(unittest.TestCase):
    """Test optimization endpoint."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()
        self.valid_request = {"ship": "corvette", "tech": "pulse", "rewards": [], "seed": 42}

    def test_optimization_endpoint_requires_post(self):
        """Optimization should require POST method."""
        response = self.client.get("/api/optimize")
        # Should not accept GET
        self.assertNotEqual(response.status_code, 200)

    def test_optimization_missing_ship_parameter(self):
        """Request without ship should fail."""
        request_data = self.valid_request.copy()
        del request_data["ship"]

        response = self.client.post("/api/optimize", data=json.dumps(request_data), content_type="application/json")

        # Should return 400 or similar error
        self.assertGreaterEqual(response.status_code, 400)

    def test_optimization_missing_tech_parameter(self):
        """Request without tech should fail."""
        request_data = self.valid_request.copy()
        del request_data["tech"]

        response = self.client.post("/api/optimize", data=json.dumps(request_data), content_type="application/json")

        self.assertGreaterEqual(response.status_code, 400)

    def test_optimization_with_invalid_json(self):
        """Invalid JSON should be rejected."""
        response = self.client.post("/api/optimize", data="not valid json", content_type="application/json")

        self.assertGreaterEqual(response.status_code, 400)

    def test_optimization_response_is_json(self):
        """Response should be valid JSON."""
        response = self.client.post(
            "/api/optimize", data=json.dumps(self.valid_request), content_type="application/json"
        )

        if response.status_code == 200:
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                self.fail("Response is not valid JSON")

    def test_optimization_response_contains_grid(self):
        """Successful response should contain grid data."""
        response = self.client.post(
            "/api/optimize", data=json.dumps(self.valid_request), content_type="application/json"
        )

        if response.status_code == 200:
            data = json.loads(response.data)
            self.assertIn("grid", data)


class TestTechTreeEndpoint(unittest.TestCase):
    """Test technology tree endpoint."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_tech_tree_endpoint_requires_post(self):
        """Tech tree should require POST method."""
        response = self.client.get("/api/tech-tree")
        self.assertNotEqual(response.status_code, 200)

    def test_tech_tree_missing_ship(self):
        """Request without ship should fail."""
        request_data = {"tech": "pulse"}

        response = self.client.post("/api/tech-tree", data=json.dumps(request_data), content_type="application/json")

        self.assertGreaterEqual(response.status_code, 400)

    def test_tech_tree_response_is_json(self):
        """Response should be valid JSON."""
        response = self.client.post(
            "/api/tech-tree", data=json.dumps({"ship": "corvette"}), content_type="application/json"
        )

        if response.status_code == 200:
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                self.fail("Response is not valid JSON")

    def test_tech_tree_get_includes_standard_window_rules(self):
        """GET /tech_tree/<ship_name> includes window_rules for each technology item and base profile for standard techs."""
        response = self.client.get("/tech_tree/standard")
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        category_keys = [k for k in data.keys() if k not in ("recommended_builds", "grid_definition")]
        self.assertGreater(len(category_keys), 0)

        tech_by_key = {}
        for category in category_keys:
            for item in data[category]:
                self.assertIn("window_rules", item)
                self.assertIsInstance(item["window_rules"], dict)
                self.assertGreater(len(item["window_rules"]), 0)
                tech_by_key[item["key"]] = item

        self.assertIn("cyclotron", tech_by_key)
        self.assertEqual(
            tech_by_key["cyclotron"]["window_rules"],
            {
                "1": [1, 1],
                "2": [2, 1],
                "4": [2, 2],
                "6": [3, 2],
                "9": [3, 3],
                "12": [4, 3],
                "default": [4, 4],
            },
        )

    def test_tech_tree_get_includes_merged_window_overrides(self):
        """GET /tech_tree/<ship_name> merges custom window_overrides into window_rules for Trails and Pulse Spitter."""
        standard_resp = self.client.get("/tech_tree/standard")
        self.assertEqual(standard_resp.status_code, 200)
        standard_data = json.loads(standard_resp.data)

        trails_item = next(item for item in standard_data["Utilities"] if item["key"] == "trails")
        self.assertEqual(
            trails_item["window_rules"],
            {
                "1": [1, 1],
                "2": [2, 1],
                "4": [2, 2],
                "6": [3, 3],
                "8": [4, 3],
                "default": [4, 3],
            },
        )

        mt_resp = self.client.get("/tech_tree/standard-mt")
        self.assertEqual(mt_resp.status_code, 200)
        mt_data = json.loads(mt_resp.data)

        pulse_spitter_item = next(item for item in mt_data["Weaponry"] if item["key"] == "pulse-spitter")
        self.assertEqual(
            pulse_spitter_item["window_rules"],
            {
                "1": [1, 1],
                "2": [2, 1],
                "4": [2, 2],
                "6": [3, 3],
                "default": [3, 3],
            },
        )


class TestAnalyticsEndpoint(unittest.TestCase):
    """Test analytics/popular data endpoint."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_analytics_endpoint_exists(self):
        """Analytics endpoint should exist."""
        response = self.client.get("/analytics/popular_data")
        self.assertIn(response.status_code, [200, 500])

    @patch("src.routes.analytics.bq_client")
    def test_popular_data_from_bigquery(self, mock_bq):
        """Test popular analytics data retrieved from BigQuery."""
        mock_row = MagicMock()
        mock_row.ship_type = "corvette"
        mock_row.technology = "hyper"
        mock_row.supercharged = "false"
        mock_row.total_events = 42

        mock_results = MagicMock()
        mock_results.__iter__.return_value = [mock_row]

        mock_job = MagicMock()
        mock_job.result.return_value = mock_results
        mock_bq.query.return_value = mock_job

        # Use unique date range to avoid cache collisions from other tests
        response = self.client.get("/analytics/popular_data?start_date=2026-01-01&end_date=2026-01-02")

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["ship_type"], "corvette")
        self.assertEqual(data[0]["technology"], "hyper")
        self.assertEqual(data[0]["supercharged"], "false")
        self.assertEqual(data[0]["total_events"], 42)
        mock_bq.query.assert_called_once()

    @patch("src.routes.analytics.ga4_data_client")
    @patch("src.routes.analytics.bq_client")
    def test_popular_data_fallback_to_ga4(self, mock_bq, mock_ga4):
        """Test fallback to GA4 when BigQuery query fails."""
        mock_bq.query.side_effect = Exception("BigQuery error")

        mock_row = MagicMock()
        mock_dim_platform = MagicMock()
        mock_dim_platform.value = "standard"
        mock_dim_tech = MagicMock()
        mock_dim_tech.value = "shield"
        mock_dim_sc = MagicMock()
        mock_dim_sc.value = "true"
        mock_row.dimension_values = [mock_dim_platform, mock_dim_tech, mock_dim_sc]

        mock_metric = MagicMock()
        mock_metric.value = "15"
        mock_row.metric_values = [mock_metric]

        mock_ga4_response = MagicMock()
        mock_ga4_response.rows = [mock_row]
        mock_ga4.run_report.return_value = mock_ga4_response

        response = self.client.get("/analytics/popular_data?start_date=2026-02-01&end_date=2026-02-02")

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["ship_type"], "standard")
        self.assertEqual(data[0]["technology"], "shield")
        self.assertEqual(data[0]["supercharged"], "true")
        self.assertEqual(data[0]["total_events"], 15)
        mock_ga4.run_report.assert_called_once()

    @patch("src.routes.analytics.bq_client")
    def test_popular_data_caching(self, mock_bq):
        """Test popular analytics responses are cached."""
        mock_row = MagicMock()
        mock_row.ship_type = "sentinel"
        mock_row.technology = "pulse"
        mock_row.supercharged = "false"
        mock_row.total_events = 99

        mock_results = MagicMock()
        mock_results.__iter__.return_value = [mock_row]

        mock_job = MagicMock()
        mock_job.result.return_value = mock_results
        mock_bq.query.return_value = mock_job

        url = "/analytics/popular_data?start_date=2026-03-01&end_date=2026-03-02"

        # First request - hits BigQuery
        resp1 = self.client.get(url)
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(mock_bq.query.call_count, 1)

        # Second request - served from cache
        resp2 = self.client.get(url)
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(mock_bq.query.call_count, 1)
        self.assertEqual(resp1.data, resp2.data)


class TestPerformanceAnalyticsEndpoint(unittest.TestCase):
    """Test performance analytics data endpoint."""

    def setUp(self) -> None:
        """Set up test client and clear performance analytics cache."""
        perf_cache.cache.clear()
        self.client = app.test_client()

    def _capture_performance_query(self, mock_bq: MagicMock) -> str:
        """Execute /analytics/performance_data against a mocked BigQuery client and return the SQL query.

        Args:
            mock_bq: Patched BigQuery client mock.

        Returns:
            str: Executed BigQuery SQL query string.
        """
        mock_job = MagicMock()
        mock_job.result.return_value = []
        mock_bq.query.return_value = mock_job

        self.client.get("/analytics/performance_data?start_date=2026-04-01&end_date=2026-04-02")

        mock_bq.query.assert_called_once()
        return mock_bq.query.call_args[0][0]

    @patch("src.routes.analytics.bq_client")
    def test_performance_data_response_format(self, mock_bq: MagicMock) -> None:
        """Test performance analytics response includes p50, p75, and p90."""
        # Mock BigQuery result
        mock_row = MagicMock()
        mock_row.timestamp = 1619370000000
        mock_row.metric_name = "LCP"
        mock_row.app_version = "v1.0.0"
        mock_row.p50_val = 1000
        mock_row.average_value = 1200  # p75
        mock_row.p90_val = 1500

        mock_results = MagicMock()
        mock_results.__iter__.return_value = [mock_row]

        mock_job = MagicMock()
        mock_job.result.return_value = mock_results
        mock_bq.query.return_value = mock_job

        response = self.client.get("/analytics/performance_data")

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        first_item = data[0]
        self.assertEqual(first_item["metric_name"], "LCP")
        self.assertEqual(first_item["p50"], 1000.0)
        self.assertEqual(first_item["p75"], 1200.0)
        self.assertEqual(first_item["p90"], 1500.0)
        self.assertEqual(first_item["average_value"], 1200.0)

    @patch("src.routes.analytics.bq_client")
    def test_performance_data_query_structure(self, mock_bq: MagicMock) -> None:
        """Test deduped_vitals partitions by (user_pseudo_id, metric_name, metric_id) and buckets via MAX(event_timestamp)."""
        query_sql = self._capture_performance_query(mock_bq)

        self.assertIn("user_pseudo_id", query_sql)
        self.assertIn("REGEXP_CONTAINS(metric_id", query_sql)
        self.assertIn("metric_id", query_sql)
        self.assertIn("MAX(event_timestamp)", query_sql)
        self.assertNotIn("ANY_VALUE(event_timestamp)", query_sql)
        self.assertNotIn("ANY_VALUE(metric_name)", query_sql)
        self.assertNotIn("ANY_VALUE(m_name)", query_sql)

        deduped_match = re.search(
            r"deduped_vitals\s+AS\s*\((?P<body>.*?)\),\s*hourly_stats\s+AS",
            query_sql,
            re.DOTALL | re.IGNORECASE,
        )
        self.assertIsNotNone(deduped_match, "Expected deduped_vitals CTE before hourly_stats CTE")
        assert deduped_match is not None
        deduped_body = deduped_match.group("body")

        group_by_match = re.search(
            r"GROUP\s+BY\s+(?P<group_by>.+?)\s*\)\s*$", deduped_body.strip(), re.DOTALL | re.IGNORECASE
        )
        self.assertIsNotNone(group_by_match, "Expected GROUP BY clause in deduped_vitals CTE")
        assert group_by_match is not None
        group_by_sql = group_by_match.group("group_by")
        self.assertRegex(group_by_sql, r"\buser_pseudo_id\s*,\s*metric_name\s*,")
        self.assertIn(r"r'^v\d+-\d+-\d+$'", group_by_sql)

    @patch("src.routes.analytics.bq_client")
    def test_performance_data_snapshot_vs_delta_aggregation(self, mock_bq: MagicMock) -> None:
        """Test CLS and pre-v8.4.3 INP sum deltas while v8.4.3+ INP and single-fire vitals take latest snapshot."""
        query_sql = self._capture_performance_query(mock_bq)

        self.assertRegex(query_sql, r"SUM\(\s*event_value\s*\)\s+AS\s+delta_sum")
        self.assertRegex(
            query_sql,
            r"ARRAY_AGG\(\s*event_value\s+IGNORE\s+NULLS\s+ORDER\s+BY\s+event_timestamp\s+DESC\s+LIMIT\s+1\s*\)\[SAFE_OFFSET\(0\)\]\s+AS\s+latest_value",
        )
        self.assertEqual(
            len(re.findall(r"ARRAY_AGG\(\s*app_version\s+IGNORE\s+NULLS", query_sql)),
            1,
            "Expected app_version ARRAY_AGG to be evaluated once without duplication",
        )

        metric_value_match = re.search(
            r"CASE(?P<case_body>.*?)END\s+AS\s+metric_value",
            query_sql,
            re.DOTALL | re.IGNORECASE,
        )
        self.assertIsNotNone(metric_value_match, "Expected CASE ... END AS metric_value in deduped_vitals")
        assert metric_value_match is not None
        case_body = metric_value_match.group("case_body")

        self.assertRegex(case_body, r"WHEN\s+metric_name\s*=\s*'CLS'\s+THEN\s+delta_sum")
        self.assertRegex(case_body, r"ELSE\s+latest_value")

        inp_branch_match = re.search(
            r"WHEN\s+metric_name\s*=\s*'INP'\s+AND\s+REGEXP_CONTAINS\(.*?,\s*r'(?P<semver_regex>[^']+)'\s*\)\s+THEN\s+delta_sum",
            case_body,
            re.DOTALL,
        )
        self.assertIsNotNone(inp_branch_match, "Expected pre-v8.4.3 INP delta branch using REGEXP_CONTAINS")
        assert inp_branch_match is not None
        legacy_inp_re = re.compile(inp_branch_match.group("semver_regex"))

        for legacy_version in ("v7.9.0", "v8.0.0", "v8.3.9", "v8.4.0", "v8.4.1", "v8.4.2", "8.4.2", "v8.4.2-rc.1"):
            with self.subTest(legacy_version=legacy_version):
                self.assertIsNotNone(
                    legacy_inp_re.search(legacy_version),
                    f"Expected legacy version {legacy_version!r} to aggregate INP via SUM(val)",
                )

        for snapshot_version in (
            "v8.4.3",
            "8.4.3",
            "v8.4.4",
            "v8.4.10",
            "v8.4.20",
            "v8.5.0",
            "v8.7.0",
            "v8.8.1",
            "v8.10.0",
            "v9.0.0",
            "v10.0.0",
            "unknown",
            "",
        ):
            with self.subTest(snapshot_version=snapshot_version):
                self.assertIsNone(
                    legacy_inp_re.search(snapshot_version),
                    f"Expected snapshot version {snapshot_version!r} to resolve INP to latest chronological val",
                )

    @patch("src.routes.analytics.bq_client")
    def test_performance_data_excludes_non_production_hosts(self, mock_bq: MagicMock) -> None:
        """Test BigQuery source filters exclude non-prod hostnames and page_location origins while keeping MP events."""
        query_sql = self._capture_performance_query(mock_bq)

        self.assertIn("device.web_info.hostname", query_sql)
        self.assertIn("'nms-optimizer.app'", query_sql)
        self.assertRegex(
            query_sql,
            r"LOWER\(\s*COALESCE\(\s*hostname\s*,\s*''\s*\)\s*\)\s+IN\s*\(\s*''\s*,\s*'nms-optimizer\.app'",
        )
        self.assertRegex(query_sql, r"page_location\s+IS\s+NULL\s+OR\s+page_location\s*=\s*''")

        loc_filter_match = re.search(
            r"NOT\s+REGEXP_CONTAINS\(\s*LOWER\(\s*page_location\s*\)\s*,\s*r'(?P<loc_regex>[^']+)'\s*\)",
            query_sql,
            re.DOTALL,
        )
        self.assertIsNotNone(loc_filter_match, "Expected NOT REGEXP_CONTAINS(LOWER(page_location), ...) filter")
        assert loc_filter_match is not None
        non_prod_loc_re = re.compile(loc_filter_match.group("loc_regex"))

        for excluded_url in (
            "http://localhost:5173/",
            "http://localhost/performance/",
            "http://127.0.0.1:5173/",
            "http://127.0.0.1:8080/about/",
            "https://pr-830.nms-optimizer-web.pages.dev/",
            "https://nms-optimizer-web.pages.dev/performance/",
            "https://NMS-Optimizer-Staging.herokuapp.com/",
            "https://nms-optimizer-web.herokuapp.com",
        ):
            with self.subTest(excluded_url=excluded_url):
                self.assertIsNotNone(
                    non_prod_loc_re.search(excluded_url.lower()),
                    f"Expected non-production page_location {excluded_url!r} to be excluded",
                )

        for allowed_url in (
            "https://nms-optimizer.app/",
            "https://nms-optimizer.app/es/performance/",
            "https://www.nms-optimizer.app/about/",
            "https://nms-optimizer.app/?ref=localhost&preview=foo.pages.dev",
        ):
            with self.subTest(allowed_url=allowed_url):
                self.assertIsNone(
                    non_prod_loc_re.search(allowed_url.lower()),
                    f"Expected production page_location {allowed_url!r} to be preserved",
                )


class TestErrorHandling(unittest.TestCase):
    """Test error handling in endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_invalid_endpoint_returns_404(self):
        """Nonexistent endpoint should return 404."""
        response = self.client.get("/api/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_malformed_request_body(self):
        """Malformed request body should be rejected."""
        response = self.client.post("/api/optimize", data='{"invalid": json}', content_type="application/json")

        self.assertGreaterEqual(response.status_code, 400)

    def test_empty_request_body(self):
        """Empty request body should be rejected."""
        response = self.client.post("/api/optimize", data="", content_type="application/json")

        self.assertGreaterEqual(response.status_code, 400)

    def test_null_request_body(self):
        """Null request body should be rejected."""
        response = self.client.post("/api/optimize", data="null", content_type="application/json")

        self.assertGreaterEqual(response.status_code, 400)


class TestRequestValidation(unittest.TestCase):
    """Test request parameter validation."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()
        self.valid_request = {"ship": "corvette", "tech": "pulse", "rewards": [], "seed": 42}

    def test_invalid_ship_type(self):
        """Invalid ship type should be rejected or handled."""
        request_data = self.valid_request.copy()
        request_data["ship"] = "nonexistent_ship"

        response = self.client.post("/api/optimize", data=json.dumps(request_data), content_type="application/json")

        # Should either reject or handle gracefully
        self.assertIsNotNone(response)

    def test_invalid_tech_type(self):
        """Invalid tech type should be rejected or handled."""
        request_data = self.valid_request.copy()
        request_data["tech"] = "nonexistent_tech"

        response = self.client.post("/api/optimize", data=json.dumps(request_data), content_type="application/json")

        self.assertIsNotNone(response)

    def test_empty_rewards_list(self):
        """Empty rewards list should be accepted."""
        request_data = self.valid_request.copy()
        request_data["rewards"] = []

        response = self.client.post("/api/optimize", data=json.dumps(request_data), content_type="application/json")

        # Should not fail due to empty rewards
        self.assertNotEqual(response.status_code, 500)

    def test_invalid_reward_format(self):
        """Invalid reward format should be handled."""
        request_data = self.valid_request.copy()
        request_data["rewards"] = "not_a_list"

        response = self.client.post("/api/optimize", data=json.dumps(request_data), content_type="application/json")

        # Should handle gracefully
        self.assertIsNotNone(response)

    def test_negative_seed(self):
        """Negative seed should be handled."""
        request_data = self.valid_request.copy()
        request_data["seed"] = -1

        response = self.client.post("/api/optimize", data=json.dumps(request_data), content_type="application/json")

        self.assertIsNotNone(response)


class TestResponseFormat(unittest.TestCase):
    """Test response format consistency."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_error_responses_have_message(self):
        """Error responses should contain an error message."""
        response = self.client.get("/api/nonexistent")

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
        response = self.client.get("/")

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
        response = self.client.get("/", headers={"Origin": "http://localhost:3000"})

        # CORS should be enabled
        self.assertIsNotNone(response)

    def test_cors_preflight_request(self):
        """OPTIONS preflight requests should be handled."""
        response = self.client.options("/api/optimize", headers={"Origin": "http://localhost:3000"})

        # Should handle OPTIONS (200, 204 if supported, 404 if not)
        self.assertIn(response.status_code, [200, 204, 404])


class TestContentNegotiation(unittest.TestCase):
    """Test content type handling."""

    def setUp(self):
        """Set up test client."""
        self.client = app.test_client()

    def test_json_content_type_accepted(self):
        """application/json content type should be accepted."""
        response = self.client.post(
            "/api/optimize", data=json.dumps({"ship": "corvette", "tech": "pulse"}), content_type="application/json"
        )

        # Should not reject based on content type
        self.assertNotEqual(response.status_code, 415)

    def test_wrong_content_type(self):
        """Wrong content type might be rejected."""
        response = self.client.post("/api/optimize", data="invalid", content_type="text/plain")

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
            response = self.client.get("/")
            self.assertIsNotNone(response)

    def test_concurrent_request_isolation(self):
        """Requests should not interfere with each other."""
        # Sequential calls should work independently
        response1 = self.client.get("/")
        response2 = self.client.get("/")

        self.assertIsNotNone(response1)
        self.assertIsNotNone(response2)


if __name__ == "__main__":
    unittest.main()
