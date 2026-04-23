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
from src.database import CaseAuditLog, FraudCase, ModelRun, User, db
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.secret_key = 'super-secret-premium-key-for-masters'

    login_manager = LoginManager()
    login_manager.login_view = 'login'
    login_manager.login_message_category = 'info'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

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
        import sqlalchemy.exc
        try:
            db.create_all()
            if not User.query.filter_by(username='admin').first():
                admin = User(username='admin', role='admin')
                admin.set_password('admin123')
                db.session.add(admin)
                db.session.commit()
                logger.info("Default admin user created (admin/admin123)")
        except sqlalchemy.exc.OperationalError as e:
            if "already exists" not in str(e).lower():
                raise

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
    @login_required
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
    @login_required
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
                                        app.config['RANDOM_STATE'],
                                        app.config['MODELS_DIR'])
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
    @login_required
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
    @login_required
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

        recent_cases = FraudCase.query.order_by(FraudCase.created_at.desc()).limit(5).all()

        return render_template(
            'dashboard.html',
            stats=stats,
            status_chart=status_chart,
            priority_chart=priority_chart,
            comparison_chart=comparison_chart,
            models_trained=models_exist(app.config['MODELS_DIR']),
            model_results=results_data,
            best_model=best_model,
            recent_cases=recent_cases,
        )

    # ── Case management routes ──────────────────────────────────────────────

    @app.route('/cases')
    @login_required
    def cases():
        status_filter = request.args.get('status', '')
        priority_filter = request.args.get('priority', '')
        page = request.args.get('page', 1, type=int)
        per_page = 10

        query = FraudCase.query
        
        if current_user.role != 'admin':
            query = query.filter_by(assigned_owner=current_user.username)
        
        if status_filter:
            query = query.filter_by(status=status_filter)
        if priority_filter:
            query = query.filter_by(priority=priority_filter)

        total_count = query.count()
        import math
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
        cases_list = query.order_by(FraudCase.created_at.desc()) \
            .paginate(page=page, per_page=per_page, error_out=False).items
        
        all_users = User.query.all()
        return render_template(
            'cases.html',
            cases=cases_list,
            statuses=case_manager.VALID_STATUSES,
            priorities=case_manager.VALID_PRIORITIES,
            status_filter=status_filter,
            priority_filter=priority_filter,
            current_page=page,
            per_page=per_page,
            total_count=total_count,
            total_pages=total_pages,
            all_users=all_users,
        )

    @app.route('/cases/<int:case_id>')
    @login_required
    def case_detail(case_id):
        fraud_case = db.get_or_404(FraudCase, case_id)
        
        if current_user.role != 'admin' and fraud_case.assigned_owner != current_user.username:
            flash('You can only view your assigned cases.', 'warning')
            return redirect(url_for('cases'))
        
        audit_logs = (
            CaseAuditLog.query
            .filter_by(case_id=case_id)
            .order_by(CaseAuditLog.timestamp.desc())
            .all()
        )
        
        all_users = User.query.all()
        
        return render_template(
            'case_detail.html',
            case=fraud_case,
            audit_logs=audit_logs,
            statuses=case_manager.VALID_STATUSES,
            priorities=case_manager.VALID_PRIORITIES,
            is_admin=current_user.role == 'admin',
            all_users=all_users,
        )

    @app.route('/cases/create', methods=['POST'])
    @login_required
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
    @login_required
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
                if current_user.role != 'admin':
                    flash('Access Denied: Only administrators can assign cases.', 'danger')
                else:
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
    @login_required
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
    @login_required
    def api_dashboard_stats():
        return jsonify(case_manager.get_dashboard_stats())

    # ── Auth & Single Inference Routes ──────────────────────────────────────

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password):
                login_user(user)
                flash('Welcome back!', 'success')
                next_page = request.args.get('next')
                return redirect(next_page or url_for('dashboard'))
            flash('Invalid username or password.', 'danger')
        return render_template('login.html')

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            confirm = request.form.get('confirm_password')
            role = request.form.get('role', 'investigator')
            department = request.form.get('department', '').strip() or None

            if not username or not password:
                flash('Username and password are required.', 'danger')
                return redirect(url_for('signup'))
            if password != confirm:
                flash('Passwords do not match.', 'danger')
                return redirect(url_for('signup'))
            if User.query.filter_by(username=username).first():
                flash('Username already exists.', 'danger')
                return redirect(url_for('signup'))

            user = User(username=username, role=role, department=department)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login'))
        return render_template('signup.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('You have been logged out.', 'info')
        return redirect(url_for('login'))

    @app.route('/settings', methods=['GET', 'POST'])
    @login_required
    def settings():
        if request.method == 'POST':
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')
            
            if not current_user.check_password(current_password):
                flash('Current password is incorrect.', 'danger')
            elif new_password != confirm_password:
                flash('New passwords do not match.', 'danger')
            elif len(new_password) < 6:
                flash('Password must be at least 6 characters long.', 'warning')
            else:
                current_user.set_password(new_password)
                db.session.commit()
                flash('Password updated successfully!', 'success')
                return redirect(url_for('settings'))
                
        return render_template('settings.html')

    @app.route('/predict_single', methods=['GET', 'POST'])
    @login_required
    def predict_single():
        from src.model_utils import models_exist
        if not models_exist(app.config['MODELS_DIR']):
            flash('Models are not trained yet. Please train them first.', 'warning')
            return redirect(url_for('train'))

        prediction = None
        if request.method == 'POST':
            try:
                # Get the comma-separated string of features
                features_str = request.form.get('features', '')
                if not features_str:
                    flash('Please provide feature values.', 'danger')
                    return redirect(url_for('predict_single'))
                
                # Convert string to list of floats
                import numpy as np
                features_list = [float(x.strip()) for x in features_str.split(',') if x.strip()]
                
                # If they pasted 31 values (including Class), drop the last one
                if len(features_list) == 31:
                    features_list = features_list[:30]
                elif len(features_list) != 30:
                    flash(f'Expected 30 features, but got {len(features_list)}. Please provide Time, V1-V28, and Amount.', 'danger')
                    return redirect(url_for('predict_single'))

                # Input order is: Time (0), V1..V28 (1..28), Amount (29)
                time_val = features_list[0]
                v_features = features_list[1:29]
                amount_val = features_list[29]

                # Load scaler and scale Amount and Time
                import joblib
                import os
                scaler_path = os.path.join(app.config['MODELS_DIR'], 'scaler.pkl')
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    scaled = scaler.transform([[amount_val, time_val]])
                    amount_scaled = scaled[0][0]
                    time_scaled = scaled[0][1]
                else:
                    # Fallback if scaler missing (though models might be inaccurate)
                    amount_scaled = amount_val
                    time_scaled = time_val
                
                # Model expects: V1..V28, Amount_scaled, Time_scaled
                final_features = v_features + [amount_scaled, time_scaled]
                features_arr = np.array(final_features).reshape(1, -1)

                model_name = request.form.get('model', 'random_forest')
                from src.model_utils import load_model
                from src.predict import predict_transaction

                model = load_model(model_name, app.config['MODELS_DIR'])
                model_type = 'keras' if model_name == 'neural_network' else 'sklearn'
                prediction = predict_transaction(model, features_arr, model_type)
                
                # Check prediction structure and adjust if necessary
                if isinstance(prediction, list) and len(prediction) > 0:
                    prediction = prediction[0]

            except ValueError as ve:
                flash(f'Value Error: {str(ve)}. Ensure all values are numerical.', 'danger')
            except Exception as e:
                flash(f'Prediction error: {str(e)}', 'danger')

        all_users = User.query.all()
        return render_template('predict_single.html', prediction=prediction, features_str=request.form.get('features', ''), all_users=all_users)

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
    app.run(debug=True, host='0.0.0.0', port=7860)
