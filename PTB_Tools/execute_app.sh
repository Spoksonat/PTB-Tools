#!/bin/sh

open -a "Google Chrome" main/app/templates/index.html
python3 -m flask --app main/run_app.py run --host=127.0.0.1 --port=5000