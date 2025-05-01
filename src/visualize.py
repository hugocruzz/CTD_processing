"""Visualization functions for CTD data."""

import plotly.subplots as sp
import plotly.graph_objects as go

def create_profile_plot(df, filename):
    """Create interactive profile plot of CTD data."""
    params = [col for col in df.columns if col not in ['Date', 'Time', 'Depth']]
    n_params = len(params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig = sp.make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=params,
        shared_yaxes=True
    )
    
    for idx, param in enumerate(params):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        fig.add_trace(
            go.Scatter(x=df[param], y=df['Depth'], name=param),
            row=row, col=col
        )
        fig.update_yaxes(autorange="reversed", title_text="Depth (m)")
    
    fig.update_layout(
        height=300*n_rows,
        width=1000,
        title_text=f"CTD Profile - {filename}",
        showlegend=False
    )
    
    return fig