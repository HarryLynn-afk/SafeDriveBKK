"""Root WSGI entry point for Render/Gunicorn."""

from src.app import create_app

app = create_app()
