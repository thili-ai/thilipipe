"""The four services behind the orchestrator.

Each is a separate uvicorn process in `run.py`. They share nothing but the `state/` directory —
which is the point: in production these would be a model registry, a feature store and a config
service, updated on cadences nobody coordinates.
"""
