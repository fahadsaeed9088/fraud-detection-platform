import json
from datetime import datetime
from flask_login import UserMixin

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='investigator')
    department = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        from werkzeug.security import generate_password_hash
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        from werkzeug.security import check_password_hash
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'role': self.role,
            'department': self.department,
        }


class FraudCase(db.Model):
    __tablename__ = 'fraud_cases'

    id = db.Column(db.Integer, primary_key=True)
    transaction_ref = db.Column(db.String(100), nullable=False)
    fraud_probability = db.Column(db.Float, nullable=False)
    predicted_label = db.Column(db.Integer, nullable=False)
    priority = db.Column(db.String(20), nullable=False, default='Medium')
    assigned_owner = db.Column(db.String(100), nullable=True)
    status = db.Column(db.String(30), nullable=False, default='New')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)

    audit_logs = db.relationship(
        'CaseAuditLog',
        backref='case',
        lazy=True,
        cascade='all, delete-orphan',
    )

    def to_dict(self):
        return {
            'id': self.id,
            'transaction_ref': self.transaction_ref,
            'fraud_probability': self.fraud_probability,
            'predicted_label': self.predicted_label,
            'priority': self.priority,
            'assigned_owner': self.assigned_owner,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'notes': self.notes,
        }


class CaseAuditLog(db.Model):
    __tablename__ = 'case_audit_log'

    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey('fraud_cases.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)
    old_value = db.Column(db.String(200), nullable=True)
    new_value = db.Column(db.String(200), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    comment = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'case_id': self.case_id,
            'action': self.action,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'comment': self.comment,
        }


class ModelRun(db.Model):
    __tablename__ = 'model_runs'

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(50), unique=True, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    all_results_json = db.Column(db.Text, nullable=True)
    eda_json = db.Column(db.Text, nullable=True)
    best_model = db.Column(db.String(50), nullable=True)
    dataset_version = db.Column(db.String(100), nullable=True)

    def get_results(self):
        if self.all_results_json:
            return json.loads(self.all_results_json)
        return {}

    def get_eda(self):
        if self.eda_json:
            return json.loads(self.eda_json)
        return {}

    def to_dict(self):
        return {
            'id': self.id,
            'run_id': self.run_id,
            'date': self.date.isoformat() if self.date else None,
            'results': self.get_results(),
            'eda': self.get_eda(),
            'best_model': self.best_model,
            'dataset_version': self.dataset_version,
        }
