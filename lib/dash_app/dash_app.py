import numpy as np
import pandas as pd

from dash import Dash, html, dash_table, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go


def run_dash_app(args, trades_history, pnl_history, returns, stats, hlocv_data):
    print('----------------------------------------')
    print('######### RUN DASH APP #########')
    print('----------------------------------------')

    print('[INFO DASH] Click on the following link to access the app')
    app = Dash(__name__)

    # Filter dates where trades occurred
    available_dates = trades_history['entry_date'].dt.date.unique()

    # Define app layout with tabs
    app.layout = html.Div([
        dcc.Tabs([
            dcc.Tab(label='Trades History', children=[
                html.Div([
                    html.Label('Select Date:'),
                    dcc.Dropdown(
                        id='trades-date-filter',
                        options=[{'label': date.strftime('%Y-%m-%d'), 'value': date} for date in available_dates],
                        value=available_dates[0],
                        placeholder="Select date"
                    ),
                ]),
                dash_table.DataTable(
                    id='trades-table',
                    columns=[{"name": i, "id": i} for i in trades_history.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px', 'color': 'white', 'backgroundColor': 'black'},
                    style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold'},
                ),
                dcc.Graph(id='first-asset-graph'),
                dcc.Graph(id='second-asset-graph')
            ], style={'backgroundColor': 'black', 'color': 'white'}),

            dcc.Tab(label='Strategy Performances', children=[
                html.Div([
                    html.H2("Performance Metrics", style={'color': 'white'}),
                    html.Table([
                        html.Tr([html.Td(key, style={'color': 'white'}), html.Td(value, style={'color': 'white'})]) 
                        for key, value in stats.items()
                    ]),
                ], style={'margin': '20px', 'color': 'white', 'backgroundColor': 'black'}),
                dcc.Graph(id='cumulative-returns-graph'),
                dcc.Graph(id='drawdown-graph'),
                dcc.Graph(id='pnl-graph'),
            ], style={'backgroundColor': 'black', 'color': 'white'}),
        ], colors={"border": "white", "primary": "gray", "background": "black"})
    ], style={'backgroundColor': 'black', 'color': 'grey'})

    # Callback to filter trades table by selected date
    @app.callback(
        Output('trades-table', 'data'),
        [Input('trades-date-filter', 'value')]
    )
    def update_trades_table(selected_date):
        filtered_trades = trades_history
        if selected_date:
            filtered_trades = filtered_trades[filtered_trades['entry_date'].dt.date == pd.to_datetime(selected_date).date()]
        return filtered_trades.to_dict('records')

    # Callback to generate log-price graphs with trade zones and specific legends
    @app.callback(
        [Output('first-asset-graph', 'figure'),
         Output('second-asset-graph', 'figure')],
        [Input('trades-date-filter', 'value')]
    )
    def update_intraday_graphs(selected_date):
        if not selected_date:
            return go.Figure(), go.Figure()

        # Filter intraday data by date and hours
        filtered_data = hlocv_data[
            (hlocv_data.index.date == pd.to_datetime(selected_date).date()) &
            (hlocv_data.index.time >= pd.to_datetime('10:00').time()) &
            (hlocv_data.index.time <= pd.to_datetime('19:00').time())
        ]

        # Separate data for the two assets
        asset1_data = filtered_data[filtered_data['ticker'] == trades_history['symbol'].unique()[0]].iloc[args.train_size:]
        asset2_data = filtered_data[filtered_data['ticker'] == trades_history['symbol'].unique()[1]].iloc[args.train_size:]

        # Create figures for log-prices
        asset1_fig, asset2_fig = go.Figure(), go.Figure()
        
        asset1_fig.add_trace(go.Scatter(x=asset1_data.index, y=np.log(asset1_data['close']),
                                        mode='lines', name=f'{trades_history["symbol"].unique()[0]} log(px_close)',
                                        line=dict(color='blue')))
        asset2_fig.add_trace(go.Scatter(x=asset2_data.index, y=np.log(asset2_data['close']),
                                        mode='lines', name=f'{trades_history["symbol"].unique()[1]} log(px_close)',
                                        line=dict(color='blue')))

        # Add trade zones and legends
        for _, trade in trades_history.iterrows():
            if pd.to_datetime(trade['entry_date']).date() == pd.to_datetime(selected_date).date():
                start_time, end_time = trade['entry_date'], trade['exit_date']
                color = 'green' if trade['size'] > 0 else 'red'
                
                if trade['symbol'] == trades_history['symbol'].unique()[0]:
                    asset1_fig.add_shape(type="rect", x0=start_time, x1=end_time,
                                         y0=min(np.log(asset1_data['close'])),
                                         y1=max(np.log(asset1_data['close'])),
                                         fillcolor=color, opacity=0.3, layer="below", line_width=0)
                if trade['symbol'] == trades_history['symbol'].unique()[1]:
                    asset2_fig.add_shape(type="rect", x0=start_time, x1=end_time,
                                         y0=min(np.log(asset2_data['close'])),
                                         y1=max(np.log(asset2_data['close'])),
                                         fillcolor=color, opacity=0.3, layer="below", line_width=0)

        # Update layout with legends
        for fig in [asset1_fig, asset2_fig]:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                     marker=dict(size=10, color='green'), name='Long Position'))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                     marker=dict(size=10, color='red'), name='Short Position'))
            fig.update_layout(template='plotly_dark')

        asset1_fig.update_layout(title=f'{trades_history["symbol"].unique()[0]} Trading Period on {selected_date}', 
                                 xaxis_title='Time', yaxis_title='Log Price')
        asset2_fig.update_layout(title=f'{trades_history["symbol"].unique()[1]} Trading Period on {selected_date}', 
                                 xaxis_title='Time', yaxis_title='Log Price')

        return asset1_fig, asset2_fig

    # Callbacks for P&L, Cumulative Returns, and Drawdown graphs
    @app.callback(
        [Output('pnl-graph', 'figure'),
         Output('cumulative-returns-graph', 'figure'),
         Output('drawdown-graph', 'figure')],
        [Input('pnl-graph', 'id')]
    )
    def update_pnl_graph(_):
        pnl_fig = go.Figure(data=[go.Bar(x=pnl_history.index, y=pnl_history, name='Significant Daily Returns')])
        pnl_fig.update_layout(title='P&L Over Time', xaxis_title='Date', yaxis_title='P&L', template='plotly_dark')

        cumulative_returns = (1 + returns).cumprod()
        cumulative_fig = go.Figure(data=[go.Scatter(x=cumulative_returns.index, y=cumulative_returns, 
                                                    mode='lines', name='Cumulative Returns', line=dict(color='green'))])
        cumulative_fig.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Return', template='plotly_dark')

        drawdown = (cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()
        drawdown_fig = go.Figure(data=[go.Scatter(x=drawdown.index, y=drawdown, 
                                                  fill='tozeroy', mode='lines', name='Drawdown', line=dict(color='red'))])
        drawdown_fig.update_layout(title='Drawdowns', xaxis_title='Date', yaxis_title='Drawdown', template='plotly_dark')

        return pnl_fig, cumulative_fig, drawdown_fig

    # Run the Dash server
    app.run_server(debug=False)
