#!/bin/bash
export GA4_API_SECRET=dEm7eUHHROOwljFuD2qrHg
gunicorn --preload --timeout 120 src.app:app --keep-alive 60 --worker-class gevent
