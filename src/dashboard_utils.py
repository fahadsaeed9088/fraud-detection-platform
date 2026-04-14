import logging

import plotly.graph_objects as go

logger = logging.getLogger(__name__)

_PALETTE = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1']


def generate_model_comparison_chart(results: dict) -> str:
    """Bar chart comparing ROC-AUC and F1-Score for all models."""
    models = list(results.keys())
    roc_aucs = [results[m]['roc_auc'] for m in models]
    f1_scores = [results[m]['f1'] for m in models]

    fig = go.Figure(data=[
        go.Bar(name='ROC-AUC', x=models, y=roc_aucs, marker_color='#4e79a7'),
        go.Bar(name='F1-Score', x=models, y=f1_scores, marker_color='#f28e2b'),
    ])
    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        yaxis=dict(range=[0, 1.05], title='Score'),
        xaxis_title='Model',
        template='plotly_white',
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig.to_json()


def generate_confusion_matrix_chart(cm: list, model_name: str) -> str:
    """Heatmap for a single model's confusion matrix."""
    labels = ['Legitimate', 'Fraud']
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True,
        text=[[str(v) for v in row] for row in cm],
        texttemplate='%{text}',
        textfont={'size': 16},
    ))
    fig.update_layout(
        title=f'Confusion Matrix — {model_name}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template='plotly_white',
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig.to_json()


def generate_case_status_chart(status_counts: dict) -> str:
    """Donut chart for fraud case statuses."""
    labels = list(status_counts.keys())
    values = list(status_counts.values())
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker_colors=_PALETTE,
    )])
    fig.update_layout(
        title='Cases by Status',
        template='plotly_white',
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig.to_json()


def generate_case_priority_chart(priority_counts: dict) -> str:
    """Bar chart for case priority distribution."""
    color_map = {'Low': '#59a14f', 'Medium': '#f28e2b', 'High': '#e15759', 'Critical': '#b07aa1'}
    priorities = list(priority_counts.keys())
    counts = list(priority_counts.values())
    colors = [color_map.get(p, '#4e79a7') for p in priorities]

    fig = go.Figure(data=[go.Bar(x=priorities, y=counts, marker_color=colors)])
    fig.update_layout(
        title='Cases by Priority',
        xaxis_title='Priority',
        yaxis_title='Count',
        template='plotly_white',
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig.to_json()


def generate_metrics_radar_chart(results: dict) -> str:
    """Radar/spider chart showing all metrics per model."""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    fig = go.Figure()
    for i, (model_name, metrics) in enumerate(results.items()):
        values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['roc_auc'],
        ]
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=model_name,
            line_color=_PALETTE[i % len(_PALETTE)],
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='Model Metrics Overview',
        template='plotly_white',
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig.to_json()
