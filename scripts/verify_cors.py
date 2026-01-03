import requests
from flask import Flask
from flask_cors import CORS
import threading
import time
import os
import re


# Mock the app for testing purposes since we can't easily spin up the full production app
# This script verifies that the flask-cors configuration logic works as intended
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

    CORS(app, resources={r"/*": {"origins": allowed_origins}}, supports_credentials=True)

    @app.route("/")
    def index():
        return "OK"

    # Run app in a thread
    port = 5050
    thread = threading.Thread(target=lambda: app.run(port=port, debug=False, use_reloader=False))
    thread.daemon = True
    thread.start()
    time.sleep(2)  # Wait for startup

    base_url = f"http://localhost:{port}"

    # Test Case 1: Allowed Origin
    print("Test 1: Allowed Origin (https://nms-optimizer.app)")
    headers = {"Origin": "https://nms-optimizer.app"}
    try:
        response = requests.get(base_url, headers=headers)
        ac_allow_origin = response.headers.get("Access-Control-Allow-Origin")
        ac_allow_creds = response.headers.get("Access-Control-Allow-Credentials")

        print(f"Status: {response.status_code}")
        print(f"Access-Control-Allow-Origin: {ac_allow_origin}")
        print(f"Access-Control-Allow-Credentials: {ac_allow_creds}")

        if ac_allow_origin == "https://nms-optimizer.app" and ac_allow_creds == "true":
            print("PASS")
        else:
            print("FAIL")

    except Exception as e:
        print(f"FAIL: {e}")

    print("-" * 20)

    # Test Case 2: Allowed Localhost (any port)
    print("Test 2: Allowed Localhost (port 4173)")
    headers = {"Origin": "http://localhost:4173"}
    try:
        response = requests.get(base_url, headers=headers)
        ac_allow_origin = response.headers.get("Access-Control-Allow-Origin")

        print(f"Access-Control-Allow-Origin: {ac_allow_origin}")
        if ac_allow_origin == "http://localhost:4173":
            print("PASS")
        else:
            print("FAIL")
    except Exception as e:
        print(f"FAIL: {e}")

    print("-" * 20)

    # Test Case 2.1: Random Localhost Port (often used in tests)
    print("Test 2.1: Random Localhost Port (port 9999)")
    headers = {"Origin": "http://localhost:9999"}
    try:
        response = requests.get(base_url, headers=headers)
        ac_allow_origin = response.headers.get("Access-Control-Allow-Origin")

        print(f"Access-Control-Allow-Origin: {ac_allow_origin}")
        if ac_allow_origin == "http://localhost:9999":
            print("PASS")
        else:
            print("FAIL")
    except Exception as e:
        print(f"FAIL: {e}")

    print("-" * 20)

    # Test Case 3: Disallowed Origin
    print("Test 3: Disallowed Origin (http://evil.com)")
    headers = {"Origin": "http://evil.com"}
    try:
        response = requests.get(base_url, headers=headers)
        ac_allow_origin = response.headers.get("Access-Control-Allow-Origin")

        print(f"Access-Control-Allow-Origin: {ac_allow_origin}")
        if ac_allow_origin is None:
            print("PASS")
        else:
            print("FAIL")
    except Exception as e:
        print(f"FAIL: {e}")


if __name__ == "__main__":
    test_mock_app()
