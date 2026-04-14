"""Shared pytest fixtures for all test modules."""
import pytest

from app import create_app
from src.database import db as _db


@pytest.fixture(scope='session')
def app():
    """Create a test Flask application with an in-memory SQLite database."""
    test_app = create_app()
    test_app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'WTF_CSRF_ENABLED': False,
        'SECRET_KEY': 'test-secret',
    })
    with test_app.app_context():
        _db.create_all()
        yield test_app
        _db.session.remove()
        _db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def db(app):
    """Yield an active database session, rolling back after each test."""
    with app.app_context():
        yield _db
        _db.session.rollback()
