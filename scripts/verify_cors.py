import requests
from flask import Flask
from flask_cors import CORS
import threading
import time
import os
import re


def test_mock_app():
    app = Flask(__name__)

    DEFAULT_ORIGINS = [
        "https://nms-optimizer.app",
        re.compile(r"https?://localhost:\d+"),
    ]
    env_origins = os.environ.get("ALLOWED_ORIGINS")
    if env_origins:
        allowed_origins = env_origins.split(",")
    else:
        allowed_origins = DEFAULT_ORIGINS

    # Split policy
    CORS(
        app,
        resources={r"/api/events": {"origins": allowed_origins, "supports_credentials": True}, r"/*": {"origins": "*"}},
    )

    @app.route("/")
    def index():
        return "OK"

    @app.route("/optimize", methods=["POST"])
    def optimize():
        return "OK"

    @app.route("/api/events", methods=["POST"])
    def events():
        return "OK"

    # Run app in a thread
    port = 5055
    thread = threading.Thread(target=lambda: app.run(port=port, debug=False, use_reloader=False))
    thread.daemon = True
    thread.start()
    time.sleep(2)  # Wait for startup

    base_url = f"http://localhost:{port}"

    # Test Case 1: Strict endpoint - Allowed Origin
    print("Test 1: Strict endpoint (/api/events) - Allowed Origin")
    headers = {"Origin": "https://nms-optimizer.app"}
    try:
        response = requests.post(f"{base_url}/api/events", headers=headers)
        ac_allow_origin = response.headers.get("Access-Control-Allow-Origin")
        ac_allow_creds = response.headers.get("Access-Control-Allow-Credentials")

        print(f"Access-Control-Allow-Origin: {ac_allow_origin}")
        print(f"Access-Control-Allow-Credentials: {ac_allow_creds}")

        if ac_allow_origin == "https://nms-optimizer.app" and ac_allow_creds == "true":
            print("PASS")
        else:
            print("FAIL")

    except Exception as e:
        print(f"FAIL: {e}")

    print("-" * 20)

    # Test Case 2: Strict endpoint - Disallowed Origin
    print("Test 2: Strict endpoint (/api/events) - Disallowed Origin")
    headers = {"Origin": "http://evil.com"}
    try:
        response = requests.post(f"{base_url}/api/events", headers=headers)
        ac_allow_origin = response.headers.get("Access-Control-Allow-Origin")

        print(f"Access-Control-Allow-Origin: {ac_allow_origin}")
        if ac_allow_origin is None:
            print("PASS")
        else:
            print("FAIL")
    except Exception as e:
        print(f"FAIL: {e}")

    print("-" * 20)

    # Test Case 3: Permissive endpoint - Any Origin
    print("Test 3: Permissive endpoint (/optimize) - Any Origin")
    headers = {"Origin": "http://storybook.internal"}
    try:
        response = requests.post(f"{base_url}/optimize", headers=headers)
        ac_allow_origin = response.headers.get("Access-Control-Allow-Origin")

        print(f"Access-Control-Allow-Origin: {ac_allow_origin}")
        if ac_allow_origin == "*" or ac_allow_origin == "http://storybook.internal":
            print("PASS")
        else:
            print("FAIL")
    except Exception as e:
        print(f"FAIL: {e}")

    print("-" * 20)

    # Test Case 4: Catch-all endpoint - Any Origin
    print("Test 4: Catch-all endpoint (/) - Any Origin")
    headers = {"Origin": "http://random.origin"}
    try:
        response = requests.get(base_url, headers=headers)
        ac_allow_origin = response.headers.get("Access-Control-Allow-Origin")

        print(f"Access-Control-Allow-Origin: {ac_allow_origin}")
        if ac_allow_origin == "*" or ac_allow_origin == "http://random.origin":
            print("PASS")
        else:
            print("FAIL")
    except Exception as e:
        print(f"FAIL: {e}")


if __name__ == "__main__":
    test_mock_app()
