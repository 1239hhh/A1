"""Convenience entry point for running EvidenceFlow from the repository root.

Run: uvicorn app:app --reload
The canonical implementation remains in app/main.py.
"""
from app.main import app

__all__ = ["app"]
