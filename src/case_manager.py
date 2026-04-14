import logging
from datetime import datetime

from src.database import CaseAuditLog, FraudCase, db

logger = logging.getLogger(__name__)

VALID_STATUSES = ['New', 'In Review', 'Escalated', 'Resolved', 'Closed']
VALID_PRIORITIES = ['Low', 'Medium', 'High', 'Critical']


def create_case(
    transaction_ref: str,
    fraud_probability: float,
    predicted_label: int,
    priority: str = 'Medium',
    assigned_owner: str = None,
    notes: str = None,
) -> FraudCase:
    """Create a new fraud investigation case with an audit entry."""
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"Invalid priority '{priority}'. Choose from: {VALID_PRIORITIES}")

    case = FraudCase(
        transaction_ref=transaction_ref,
        fraud_probability=round(float(fraud_probability), 4),
        predicted_label=int(predicted_label),
        priority=priority,
        assigned_owner=assigned_owner or None,
        status='New',
        notes=notes or None,
    )
    db.session.add(case)
    db.session.flush()  # obtain case.id before the log

    log = CaseAuditLog(
        case_id=case.id,
        action='CASE_CREATED',
        old_value=None,
        new_value='New',
        comment=f'Case created for transaction {transaction_ref}',
    )
    db.session.add(log)
    db.session.commit()
    logger.info("Case #%s created for transaction %s", case.id, transaction_ref)
    return case


def update_case_status(case_id: int, new_status: str, comment: str = None) -> FraudCase:
    """Transition a case to a new status and log the change."""
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{new_status}'. Choose from: {VALID_STATUSES}")

    case = db.get_or_404(FraudCase, case_id)
    old_status = case.status
    case.status = new_status
    case.updated_at = datetime.utcnow()

    log = CaseAuditLog(
        case_id=case_id,
        action='STATUS_CHANGED',
        old_value=old_status,
        new_value=new_status,
        comment=comment,
    )
    db.session.add(log)
    db.session.commit()
    logger.info("Case #%s status: %s → %s", case_id, old_status, new_status)
    return case


def assign_case(case_id: int, owner: str, comment: str = None) -> FraudCase:
    """Assign (or reassign) a case to an analyst."""
    case = db.get_or_404(FraudCase, case_id)
    old_owner = case.assigned_owner
    case.assigned_owner = owner
    case.updated_at = datetime.utcnow()

    log = CaseAuditLog(
        case_id=case_id,
        action='CASE_ASSIGNED',
        old_value=old_owner,
        new_value=owner,
        comment=comment,
    )
    db.session.add(log)
    db.session.commit()
    return case


def add_case_note(case_id: int, note: str) -> FraudCase:
    """Append a timestamped note to a case."""
    case = db.get_or_404(FraudCase, case_id)
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    existing = case.notes or ''
    sep = '\n' if existing else ''
    case.notes = f"{existing}{sep}[{ts}] {note}"
    case.updated_at = datetime.utcnow()

    log = CaseAuditLog(
        case_id=case_id,
        action='NOTE_ADDED',
        old_value=None,
        new_value=note[:200],
        comment=None,
    )
    db.session.add(log)
    db.session.commit()
    return case


def get_dashboard_stats() -> dict:
    """Aggregate statistics for the operations dashboard."""
    total = FraudCase.query.count()

    status_counts = {s: FraudCase.query.filter_by(status=s).count() for s in VALID_STATUSES}
    priority_counts = {p: FraudCase.query.filter_by(priority=p).count() for p in VALID_PRIORITIES}

    avg_raw = db.session.query(db.func.avg(FraudCase.fraud_probability)).scalar()
    avg_prob = round(float(avg_raw), 4) if avg_raw is not None else 0.0

    open_cases = (
        status_counts.get('New', 0)
        + status_counts.get('In Review', 0)
        + status_counts.get('Escalated', 0)
    )

    return {
        'total_cases': total,
        'open_cases': open_cases,
        'status_counts': status_counts,
        'priority_counts': priority_counts,
        'avg_fraud_probability': avg_prob,
    }
