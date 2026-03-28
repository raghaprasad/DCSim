#!/bin/bash
# Build the dcsim wheel for Pyodide consumption.
# Run from repo root: bash web/build_wheel.sh
set -e
pip wheel --no-deps -w web/dist/ .
echo "Wheel built: $(ls web/dist/*.whl)"
