"""Tests for the fraud case workflow module."""
import pytest

from src import case_manager
from src.database import CaseAuditLog, FraudCase


class TestCreateCase:
    def test_creates_case_with_correct_fields(self, app, db):
        with app.app_context():
            case = case_manager.create_case(
                transaction_ref='TXN-001',
                fraud_probability=0.85,
                predicted_label=1,
                priority='High',
                assigned_owner='Alice',
                notes='Suspicious pattern detected.',
            )
            assert case.id is not None
            assert case.transaction_ref == 'TXN-001'
            assert case.fraud_probability == 0.85
            assert case.priority == 'High'
            assert case.assigned_owner == 'Alice'
            assert case.status == 'New'

    def test_invalid_priority_raises(self, app, db):
        with app.app_context():
            with pytest.raises(ValueError, match="Invalid priority"):
                case_manager.create_case(
                    transaction_ref='TXN-002',
                    fraud_probability=0.5,
                    predicted_label=1,
                    priority='CRITICAL_PLUS',
                )

    def test_creation_generates_audit_log(self, app, db):
        with app.app_context():
            case = case_manager.create_case(
                transaction_ref='TXN-003',
                fraud_probability=0.7,
                predicted_label=1,
            )
            logs = CaseAuditLog.query.filter_by(case_id=case.id).all()
            assert len(logs) >= 1
            assert logs[0].action == 'CASE_CREATED'


class TestUpdateCaseStatus:
    def _make_case(self, app):
        with app.app_context():
            return case_manager.create_case(
                transaction_ref='TXN-STATUS',
                fraud_probability=0.9,
                predicted_label=1,
            )

    def test_status_update_saves_correctly(self, app, db):
        with app.app_context():
            case = case_manager.create_case('TXN-S1', 0.9, 1)
            updated = case_manager.update_case_status(case.id, 'In Review', 'Started review.')
            assert updated.status == 'In Review'

    def test_status_update_creates_audit_log(self, app, db):
        with app.app_context():
            case = case_manager.create_case('TXN-S2', 0.9, 1)
            case_manager.update_case_status(case.id, 'Escalated', 'Needs senior review.')
            logs = CaseAuditLog.query.filter_by(case_id=case.id, action='STATUS_CHANGED').all()
            assert len(logs) == 1
            assert logs[0].old_value == 'New'
            assert logs[0].new_value == 'Escalated'

    def test_invalid_status_raises(self, app, db):
        with app.app_context():
            case = case_manager.create_case('TXN-S3', 0.9, 1)
            with pytest.raises(ValueError, match="Invalid status"):
                case_manager.update_case_status(case.id, 'UNKNOWN_STATUS')


class TestAssignCase:
    def test_assign_saves_owner(self, app, db):
        with app.app_context():
            case = case_manager.create_case('TXN-A1', 0.75, 1)
            updated = case_manager.assign_case(case.id, 'Bob')
            assert updated.assigned_owner == 'Bob'

    def test_reassign_generates_audit(self, app, db):
        with app.app_context():
            case = case_manager.create_case('TXN-A2', 0.8, 1, assigned_owner='Alice')
            case_manager.assign_case(case.id, 'Charlie', 'Workload rebalancing.')
            logs = CaseAuditLog.query.filter_by(case_id=case.id, action='CASE_ASSIGNED').all()
            assert len(logs) == 1
            assert logs[0].old_value == 'Alice'
            assert logs[0].new_value == 'Charlie'


class TestAddCaseNote:
    def test_note_appended(self, app, db):
        with app.app_context():
            case = case_manager.create_case('TXN-N1', 0.6, 1)
            updated = case_manager.add_case_note(case.id, 'Investigated — no chargeback.')
            assert 'Investigated — no chargeback.' in updated.notes

    def test_multiple_notes_all_stored(self, app, db):
        with app.app_context():
            case = case_manager.create_case('TXN-N2', 0.6, 1)
            case_manager.add_case_note(case.id, 'First note.')
            updated = case_manager.add_case_note(case.id, 'Second note.')
            assert 'First note.' in updated.notes
            assert 'Second note.' in updated.notes

    def test_note_generates_audit(self, app, db):
        with app.app_context():
            case = case_manager.create_case('TXN-N3', 0.6, 1)
            case_manager.add_case_note(case.id, 'Audit note test.')
            logs = CaseAuditLog.query.filter_by(case_id=case.id, action='NOTE_ADDED').all()
            assert len(logs) == 1


class TestDashboardStats:
    def test_stats_keys_present(self, app, db):
        with app.app_context():
            stats = case_manager.get_dashboard_stats()
            for key in ['total_cases', 'open_cases', 'status_counts',
                        'priority_counts', 'avg_fraud_probability']:
                assert key in stats

    def test_total_cases_increments(self, app, db):
        with app.app_context():
            before = case_manager.get_dashboard_stats()['total_cases']
            case_manager.create_case('TXN-STAT', 0.5, 1)
            after = case_manager.get_dashboard_stats()['total_cases']
            assert after == before + 1
