import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_results(results, title='Pairs Trading'):
    """Three-panel plot: equity curve, spread, positions."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.35, 0.2],
        subplot_titles=['Equity Curve', 'Spread', 'Position']
    )

    fig.add_trace(go.Scatter(
        x=results.index, y=results['equity'],
        mode='lines', name='Equity',
        line=dict(color='#00b4d8', width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=results.index, y=results['spread'],
        mode='lines', name='Spread',
        line=dict(color='#90e0ef', width=1)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=results.index, y=results['position'],
        mode='lines', name='Position',
        line=dict(color='#f77f00', width=1)
    ), row=3, col=1)

    fig.update_layout(
        template='plotly_dark', title=title,
        height=700, showlegend=True
    )
    fig.show()
