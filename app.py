import json
import logging
import os
import uuid
from datetime import datetime

from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   session, url_for)
from werkzeug.utils import secure_filename

from config import Config
from src import case_manager
from src.database import CaseAuditLog, FraudCase, ModelRun, db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure required directories exist at startup
    for d in [
        config_class.DATA_RAW_DIR,
        config_class.DATA_PROCESSED_DIR,
        config_class.MODELS_DIR,
        os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance'),
    ]:
        os.makedirs(d, exist_ok=True)

    db.init_app(app)

    with app.app_context():
        db.create_all()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _allowed_file(filename: str) -> bool:
        return (
            '.' in filename
            and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
        )

    def _dataset_path() -> str:
        return os.path.join(app.config['DATA_RAW_DIR'], 'creditcard.csv')

    def _load_latest_results():
        """Fetch the most recent training run results from the database."""
        run = ModelRun.query.order_by(ModelRun.date.desc()).first()
        if run:
            return run.get_results(), run.get_eda(), run.best_model
        return None, None, None

    # ── Core routes ────────────────────────────────────────────────────────

    @app.route('/')
    def index():
        from src.model_utils import models_exist
        stats = None
        try:
            stats = case_manager.get_dashboard_stats()
        except Exception:
            pass
        dataset_ready = os.path.exists(_dataset_path())
        return render_template(
            'index.html',
            models_trained=models_exist(app.config['MODELS_DIR']),
            stats=stats,
            dataset_ready=dataset_ready,
        )

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part in the request.', 'danger')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('No file selected.', 'danger')
                return redirect(request.url)

            if not _allowed_file(file.filename):
                flash('Only CSV files are accepted.', 'danger')
                return redirect(request.url)

            filepath = _dataset_path()
            file.save(filepath)

            try:
                from src.preprocessing import generate_eda_summary, load_data, validate_columns
                df = load_data(filepath)
                validate_columns(df)
                eda = generate_eda_summary(df)
                flash('Dataset uploaded and validated successfully!', 'success')
                return render_template('upload.html', eda=eda, uploaded=True)
            except Exception as exc:
                os.remove(filepath)
                flash(f'Validation error: {exc}', 'danger')
                return redirect(request.url)

        uploaded = os.path.exists(_dataset_path())
        eda = None
        if uploaded:
            try:
                from src.preprocessing import generate_eda_summary, load_data
                eda = generate_eda_summary(load_data(_dataset_path()))
            except Exception:
                pass
        return render_template('upload.html', eda=eda, uploaded=uploaded)

    @app.route('/train', methods=['GET', 'POST'])
    def train():
        if request.method == 'POST':
            filepath = _dataset_path()
            if not os.path.exists(filepath):
                flash('Please upload the dataset first.', 'danger')
                return redirect(url_for('upload'))

            try:
                from src.evaluate_models import (evaluate_all_models,
                                                 get_best_model)
                from src.preprocessing import run_full_pipeline
                from src.train_models import train_all_models

                data = run_full_pipeline(filepath, app.config['TEST_SIZE'],
                                        app.config['RANDOM_STATE'])
                models = train_all_models(
                    data['X_train'], data['y_train'],
                    random_state=app.config['RANDOM_STATE'],
                    models_dir=app.config['MODELS_DIR'],
                )
                results = evaluate_all_models(models, data['X_test'], data['y_test'])
                best = get_best_model(results)

                run = ModelRun(
                    run_id=str(uuid.uuid4())[:12],
                    all_results_json=json.dumps(results),
                    eda_json=json.dumps(data['eda_summary']),
                    best_model=best,
                    dataset_version=datetime.utcnow().strftime('%Y%m%d_%H%M%S'),
                )
                db.session.add(run)
                db.session.commit()

                # Auto-create fraud cases from best model's test-set predictions
                _auto_create_cases(models, best, data['X_test'], run.run_id, max_cases=len(data['X_test']))

                flash(
                    f'All models trained and evaluated successfully! '
                    f'Fraud cases auto-generated — check Cases.',
                    'success',
                )
                return redirect(url_for('results'))

            except Exception as exc:
                logger.exception("Training error")
                flash(f'Training failed: {exc}', 'danger')
                return render_template('training.html', dataset_exists=True, error=str(exc))

        dataset_exists = os.path.exists(_dataset_path())
        return render_template('training.html', dataset_exists=dataset_exists)

    # ── Helper: auto-create cases after training ──────────────────────────

    def _auto_create_cases(models: dict, best_display: str, X_test, run_id: str,
                           max_cases: int = 100) -> None:
        """Generate FraudCase rows for high-probability fraud predictions.

        Transaction refs are based on the stable original dataset row index so
        that retraining on the same dataset never produces duplicate cases.
        Predictions are sorted by probability descending so the highest-
        confidence cases are always within the cap, never discarded in favour
        of lower-confidence ones.
        """
        _DISPLAY_TO_KEY = {
            'Logistic Regression': 'logistic_regression',
            'Random Forest': 'random_forest',
            'XGBoost': 'xgboost',
            'Neural Network': 'neural_network',
        }
        best_key = _DISPLAY_TO_KEY.get(best_display, 'random_forest')
        model = models.get(best_key) or next(iter(models.values()))
        model_type = 'keras' if best_key == 'neural_network' else 'sklearn'

        from src.database import FraudCase
        from src.predict import predict_batch

        # Preserve original CSV row indices (stable across retrains on same data)
        original_indices = list(X_test.index) if hasattr(X_test, 'index') else list(range(len(X_test)))
        X_arr = X_test.values if hasattr(X_test, 'values') else X_test
        predictions = predict_batch(model, X_arr, model_type)

        # Pair each prediction with its stable row index
        fraud_preds = [
            (original_indices[i], p)
            for i, p in enumerate(predictions)
            if p['is_fraud']
        ]

        # Sort highest probability first — ensures the cap always keeps the
        # most critical cases and never discards them for lower-confidence ones
        fraud_preds.sort(key=lambda x: x[1]['probability'], reverse=True)
        fraud_preds = fraud_preds[:max_cases]

        created = skipped = 0
        for row_idx, pred in fraud_preds:
            prob = pred['probability']
            # Stable ref: same row = same ref regardless of which run flagged it
            ref = f'DATASET-ROW{row_idx}'

            # Skip if already exists — prevents duplicate entries on retrain
            exists = db.session.query(
                FraudCase.query.filter_by(transaction_ref=ref).exists()
            ).scalar()
            if exists:
                skipped += 1
                continue

            priority = (
                'Critical' if prob >= 0.90 else
                'High'     if prob >= 0.70 else
                'Medium'
            )
            case_manager.create_case(
                transaction_ref=ref,
                fraud_probability=prob,
                predicted_label=pred['prediction'],
                priority=priority,
            )
            created += 1

        logger.info(
            "Auto-cases from run %s: %d created, %d skipped (already existed)",
            run_id, created, skipped,
        )

    @app.route('/results')
    def results():
        results_data, eda, best_model = _load_latest_results()
        if not results_data:
            flash('No training results found. Please train the models first.', 'warning')
            return redirect(url_for('train'))

        from src.dashboard_utils import (generate_confusion_matrix_chart,
                                         generate_metrics_radar_chart,
                                         generate_model_comparison_chart)

        comparison_chart = generate_model_comparison_chart(results_data)
        radar_chart = generate_metrics_radar_chart(results_data)
        cm_charts = {
            name: generate_confusion_matrix_chart(metrics['confusion_matrix'], name)
            for name, metrics in results_data.items()
        }

        return render_template(
            'results.html',
            results=results_data,
            best_model=best_model,
            comparison_chart=comparison_chart,
            radar_chart=radar_chart,
            cm_charts=cm_charts,
            eda=eda,
        )

    @app.route('/dashboard')
    def dashboard():
        from src.dashboard_utils import (generate_case_priority_chart,
                                         generate_case_status_chart,
                                         generate_model_comparison_chart)
        from src.model_utils import models_exist

        stats = case_manager.get_dashboard_stats()
        status_chart = generate_case_status_chart(stats['status_counts'])
        priority_chart = generate_case_priority_chart(stats['priority_counts'])

        results_data, _, best_model = _load_latest_results()
        comparison_chart = generate_model_comparison_chart(results_data) if results_data else None

        return render_template(
            'dashboard.html',
            stats=stats,
            status_chart=status_chart,
            priority_chart=priority_chart,
            comparison_chart=comparison_chart,
            models_trained=models_exist(app.config['MODELS_DIR']),
            model_results=results_data,
            best_model=best_model,
        )

    # ── Case management routes ──────────────────────────────────────────────

    @app.route('/cases')
    def cases():
        status_filter = request.args.get('status', '')
        priority_filter = request.args.get('priority', '')

        query = FraudCase.query
        if status_filter:
            query = query.filter_by(status=status_filter)
        if priority_filter:
            query = query.filter_by(priority=priority_filter)

        cases_list = query.order_by(FraudCase.created_at.desc()).all()
        return render_template(
            'cases.html',
            cases=cases_list,
            statuses=case_manager.VALID_STATUSES,
            priorities=case_manager.VALID_PRIORITIES,
            status_filter=status_filter,
            priority_filter=priority_filter,
        )

    @app.route('/cases/<int:case_id>')
    def case_detail(case_id):
        fraud_case = db.get_or_404(FraudCase, case_id)
        audit_logs = (
            CaseAuditLog.query
            .filter_by(case_id=case_id)
            .order_by(CaseAuditLog.timestamp.desc())
            .all()
        )
        return render_template(
            'case_detail.html',
            case=fraud_case,
            audit_logs=audit_logs,
            statuses=case_manager.VALID_STATUSES,
            priorities=case_manager.VALID_PRIORITIES,
        )

    @app.route('/cases/create', methods=['POST'])
    def create_case():
        try:
            fraud_case = case_manager.create_case(
                transaction_ref=request.form['transaction_ref'],
                fraud_probability=float(request.form['fraud_probability']),
                predicted_label=int(request.form['predicted_label']),
                priority=request.form.get('priority', 'Medium'),
                assigned_owner=request.form.get('assigned_owner') or None,
                notes=request.form.get('notes') or None,
            )
            flash(f'Case #{fraud_case.id} created successfully!', 'success')
            return redirect(url_for('case_detail', case_id=fraud_case.id))
        except Exception as exc:
            flash(f'Error creating case: {exc}', 'danger')
            return redirect(url_for('cases'))

    @app.route('/cases/update/<int:case_id>', methods=['POST'])
    def update_case(case_id):
        action = request.form.get('action', '')
        try:
            if action == 'update_status':
                case_manager.update_case_status(
                    case_id,
                    request.form['status'],
                    request.form.get('comment') or None,
                )
                flash('Status updated.', 'success')

            elif action == 'assign':
                owner = request.form.get('owner', '').strip()
                if owner:
                    case_manager.assign_case(case_id, owner, request.form.get('comment') or None)
                    flash('Case assigned.', 'success')
                else:
                    flash('Owner name cannot be empty.', 'warning')

            elif action == 'add_note':
                note = request.form.get('note', '').strip()
                if note:
                    case_manager.add_case_note(case_id, note)
                    flash('Note added.', 'success')
                else:
                    flash('Note cannot be empty.', 'warning')

            elif action == 'update_priority':
                new_priority = request.form.get('priority', '')
                if new_priority not in case_manager.VALID_PRIORITIES:
                    flash('Invalid priority value.', 'danger')
                else:
                    fraud_case = db.get_or_404(FraudCase, case_id)
                    old_priority = fraud_case.priority
                    fraud_case.priority = new_priority
                    fraud_case.updated_at = datetime.utcnow()
                    log = CaseAuditLog(
                        case_id=case_id,
                        action='PRIORITY_CHANGED',
                        old_value=old_priority,
                        new_value=new_priority,
                        comment=request.form.get('comment') or None,
                    )
                    db.session.add(log)
                    db.session.commit()
                    flash('Priority updated.', 'success')

            else:
                flash(f'Unknown action: {action}', 'warning')

        except Exception as exc:
            flash(f'Error: {exc}', 'danger')

        return redirect(url_for('case_detail', case_id=case_id))

    # ── API routes ──────────────────────────────────────────────────────────

    @app.route('/api/cases')
    def api_cases():
        cases_list = FraudCase.query.order_by(FraudCase.created_at.desc()).all()
        return jsonify([c.to_dict() for c in cases_list])

    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        payload = request.get_json(silent=True)
        if not payload or 'features' not in payload:
            return jsonify({'error': 'Request must include a "features" array.'}), 400

        model_name = payload.get('model', 'random_forest')
        try:
            import numpy as np

            from src.model_utils import load_model
            from src.predict import predict_transaction

            model = load_model(model_name, app.config['MODELS_DIR'])
            features = np.array(payload['features'], dtype=float).reshape(1, -1)
            model_type = 'keras' if model_name == 'neural_network' else 'sklearn'
            result = predict_transaction(model, features, model_type)
            return jsonify(result)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 500

    @app.route('/api/dashboard/stats')
    def api_dashboard_stats():
        return jsonify(case_manager.get_dashboard_stats())

    # ── Error handlers ──────────────────────────────────────────────────────

    @app.errorhandler(404)
    def not_found(exc):
        return render_template('error.html', error_code=404, message='Page not found.'), 404

    @app.errorhandler(500)
    def internal_error(exc):
        return render_template('error.html', error_code=500,
                               message='An internal server error occurred.'), 500

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
