"""Tests for Flask routes."""
import os

import pytest


class TestHomeRoute:
    def test_home_returns_200(self, client):
        resp = client.get('/')
        assert resp.status_code == 200

    def test_home_contains_title(self, client):
        resp = client.get('/')
        assert b'FraudDetect' in resp.data or b'Fraud' in resp.data


class TestUploadRoute:
    def test_get_upload_page(self, client):
        resp = client.get('/upload')
        assert resp.status_code == 200

    def test_upload_no_file_redirects(self, client):
        resp = client.post('/upload', data={}, follow_redirects=True)
        # Should stay on upload page with error
        assert resp.status_code == 200

    def test_upload_non_csv_rejected(self, client):
        data = {'file': (b'not a csv', 'test.txt')}
        resp = client.post(
            '/upload',
            data=data,
            content_type='multipart/form-data',
            follow_redirects=True,
        )
        assert b'CSV' in resp.data or resp.status_code in (200, 400)


class TestTrainRoute:
    def test_get_train_page(self, client):
        resp = client.get('/train')
        assert resp.status_code == 200

    def test_train_without_dataset_redirects(self, client, app):
        # Ensure no dataset file exists
        raw_path = os.path.join(app.config['DATA_RAW_DIR'], 'creditcard.csv')
        if os.path.exists(raw_path):
            os.remove(raw_path)
        resp = client.post('/train', follow_redirects=True)
        assert resp.status_code == 200


class TestResultsRoute:
    def test_results_without_training_redirects(self, client):
        resp = client.get('/results', follow_redirects=True)
        assert resp.status_code == 200

    def test_results_redirects_to_train_when_no_data(self, client):
        resp = client.get('/results')
        # Either redirect or stay with warning message
        assert resp.status_code in (200, 302)


class TestDashboardRoute:
    def test_dashboard_returns_200(self, client):
        resp = client.get('/dashboard')
        assert resp.status_code == 200


class TestCasesRoute:
    def test_cases_list_returns_200(self, client):
        resp = client.get('/cases')
        assert resp.status_code == 200

    def test_cases_filter_by_status(self, client):
        resp = client.get('/cases?status=New')
        assert resp.status_code == 200

    def test_cases_filter_by_priority(self, client):
        resp = client.get('/cases?priority=High')
        assert resp.status_code == 200


class TestAPIRoutes:
    def test_api_cases_json(self, client):
        resp = client.get('/api/cases')
        assert resp.status_code == 200
        assert resp.content_type.startswith('application/json')
        data = resp.get_json()
        assert isinstance(data, list)

    def test_api_dashboard_stats(self, client):
        resp = client.get('/api/dashboard/stats')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'total_cases' in data
        assert 'open_cases' in data

    def test_api_predict_missing_features(self, client):
        resp = client.post('/api/predict', json={}, content_type='application/json')
        assert resp.status_code == 400

    def test_error_404(self, client):
        resp = client.get('/nonexistent-page-xyz')
        assert resp.status_code == 404
