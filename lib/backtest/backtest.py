import backtrader as bt

import numpy as np
import pandas as pd

from typing import Tuple


### Backtesting Engine ###


def run_bt(hlocv_data: pd.DataFrame, params: dict, symbols: Tuple[str, str], strategy: bt.Strategy, cash: float, comm_pct: float, analyzers_config: dict) -> bt.Strategy:
    """
    Executes a backtesting simulation using the Backtrader framework.
    """
    engine = bt.Cerebro()

    for symbol in symbols:
        engine.adddata(get_data_feeds(hlocv_data, symbol), name=symbol)

    engine.addstrategy(strategy, **params)
    engine.broker.setcash(cash)
    engine.broker.setcommission(commission=comm_pct)

    for analyzer, analyzer_params in analyzers_config.items():
        engine.addanalyzer(analyzer, **analyzer_params)

    results = engine.run()
    pf_val = engine.broker.get_value()
    return results[0], pf_val


### DataFeeds ###


class CustomFeeds(bt.feeds.PandasData):
    # Custom HLOCV Feeds for Backtrader
    lines = ('datetime',)
    params = (
        ('datetime', None),  # No need because DateTimeIndex
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None)
    )


def get_data_feeds(hlocv_data: pd.DataFrame, symbol: str) -> CustomFeeds:
    """
    Filters HLOCV data for a specific symbol and prepares it for Backtrader.
    """
    hlocv_asset = hlocv_data[hlocv_data['ticker'].str.contains(symbol)]
    hlocv_asset.index = pd.to_datetime(hlocv_asset.index)
    return CustomFeeds(dataname=hlocv_asset)


### Strategy ###


class SpreadIntraday(bt.Indicator):
    lines = ('spread',)

    def __init__(self, beta, const, warmup_period):
        self.beta = beta
        self.const = const
        self.addminperiod(warmup_period)

    def compute_spread(self):
        current_logprices = np.log([self.data0[0], self.data1[0]]) 
        return current_logprices @ np.array([1.0, self.beta]) + self.const

    def next(self):
        spread = self.compute_spread()
        self.lines.spread[0] = spread


class PairTradingIntraday(bt.Strategy):
    params = dict(
        spread_std=None,
        spread_mean=None,
        beta=None,
        const=None,
        entry_factor=None,  # Multiplier for std_dev to set the entry threshold
        sl_factor=None,
        warmup_period=None,
    )

    def __init__(self):
        # Instantiate indicators
        self.spread_indicator = SpreadIntraday(self.datas[0], self.datas[1], beta=self.p.beta, const=self.p.const, warmup_period=self.p.warmup_period)
        self.upper_threshold = self.p.spread_mean + self.p.spread_std * self.p.entry_factor
        self.lower_threshold = self.p.spread_mean - self.p.spread_std * self.p.entry_factor
        # Trackers
        self.position_opened = False
        self.sl_triggered = False
        # Stop loss
        self.sl_upper = self.p.spread_mean + self.p.spread_std * self.p.sl_factor * self.p.entry_factor
        self.sl_lower = self.p.spread_mean - self.p.spread_std * self.p.sl_factor * self.p.entry_factor


    def log(self, txt, dt=None, log_type='ORDER'):
        if isinstance(dt, str):
            print(f'[INFO {log_type.upper()}] {dt} {txt}')
        else:
            dt = dt or self.datas[0].datetime.datetime(0)
            formatted_time = dt.strftime('%H:%M:%S')
            print(f'[INFO {log_type.upper()}] {formatted_time} {txt}')

    
    def notify_trade(self, trade):
        if trade.isclosed:
            entry_time = bt.num2date(trade.dtopen).strftime('%H:%M:%S')
            exit_time = bt.num2date(trade.dtclose).strftime('%H:%M:%S')
            symbol = trade.data._name
            self.log(f'CLOSED: {symbol} | Entry: {entry_time}, Exit: {exit_time} | '
                    f'Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f} | '
                    f'Commission: {trade.commission:.2f}', log_type='TRADE')


    def compute_sizes(self):
        beta = abs(self.p.beta)
        pf_val0 = 1 / (1 + beta) * self.broker.get_value()
        pf_val1 = beta / (1 + beta) * self.broker.get_value()
        size0 = int(pf_val0 / self.datas[0].close[0])
        size1 = int(pf_val1 / self.datas[1].close[0])
        return size0, size1

    def next(self):
        # Skip trading during the warmup period
        if len(self) < self.p.warmup_period:
            return

        size0, size1 = self.compute_sizes()
        position0 = self.getposition(self.datas[0]).size
        position1 = self.getposition(self.datas[1]).size

        if not self.position_opened:  # Ensure only one position is open at a time
            if all(self.spread_indicator.spread[i] > self.upper_threshold for i in range(0, 1)):
                self.sell(data=self.datas[0], size=size0)
                self.buy(data=self.datas[1], size=size1)
                self.position_opened = 'short_spread'
                self.log(f'SHORT SPREAD: Sell {self.datas[0]._name} ({-size0}) | Buy {self.datas[1]._name} ({size1})', log_type='SIGNAL')

            elif all(self.spread_indicator.spread[i] < self.lower_threshold for i in range(0, 1)):
                self.sell(data=self.datas[1], size=size1)
                self.buy(data=self.datas[0], size=size0)
                self.position_opened = 'long_spread'
                self.log(f'LONG SPREAD: Sell {self.datas[1]._name} ({-size1}) | Buy {self.datas[0]._name} ({size0})', log_type='SIGNAL')

        elif self.position_opened and not self.sl_triggered:  # Verify if a position is currently open
            mean_reverting = np.sign(self.spread_indicator.spread[0]) != np.sign(self.spread_indicator.spread[-1])
            stop_loss = self.spread_indicator.spread[0] > self.sl_upper or self.spread_indicator.spread[0] < self.sl_lower
            if mean_reverting:
                self.close(data=self.datas[0], size=abs(position0))
                self.close(data=self.datas[1], size=abs(position1))
                self.position_opened = False
                self.log(f'MEAN REVERSION: Close Position {self.datas[0]._name} ({-position0}) | Close Position {self.datas[1]._name} ({-position1})', log_type='SIGNAL')
            elif stop_loss:
                self.close(data=self.datas[0], size=abs(position0))
                self.close(data=self.datas[1], size=abs(position1))                    
                self.log(f'STOP LOSS: Close Position {self.datas[0]._name} ({-position0}) | Close Position {self.datas[1]._name} ({-position1})', log_type='SIGNAL')
                self.sl_triggered = True
                self.position_opened = 'sl_triggered'

    def stop(self):
        for data in self.datas:
            position = self.getposition(data).size
            if position != 0:  # If there's an open position
                self.close(data=data)  # Force close the position
                symbol = data._name
                self.log(f'FORCED CLOSE (End of Day): {symbol} | Size: {position}', log_type='TRADE')



### Analyzers ###


class OrdersAnalyzer(bt.Analyzer):
    def start(self):
        self.orders = []

    def notify_order(self, order):
        symbol = order.data._name
        if order.status == order.Completed:
            self.orders.append({
                'date': self.data.num2date(order.executed.dt),
                'symbol': symbol,
                'size': order.executed.size,
                'price': order.executed.price,
                'cost': order.executed.value,
                'comm': order.executed.comm,
            })

    def stop(self):
        orders = pd.DataFrame(self.orders)
        self.rets['orders'] = orders

    def get_analysis(self):
        return self.rets['orders']
    

class TradesAnalyzer(bt.Analyzer):
    def start(self):
        self.trade_sizes = {}
        self.trades = []

    def notify_trade(self, trade):
        if trade.isopen:
            self.trade_sizes[trade.ref] = trade.size
        elif trade.isclosed:
            entry_time = bt.num2date(trade.dtopen)
            exit_time = bt.num2date(trade.dtclose)
            symbol = trade.data._name
            self.trades.append({
                'symbol': symbol,
                'entry_date': entry_time,
                'exit_date': exit_time,
                'size': self.trade_sizes.get(trade.ref),
                'entry_price': trade.price,
                'exit_price': trade.data.close[0],
                'pnl_gross': trade.pnl,
                'pnl_net': trade.pnlcomm,
                'comm': trade.commission
            })

    def stop(self):
        trades = pd.DataFrame(self.trades)
        self.rets['trades'] = trades

    def get_analysis(self):
        return self.rets['trades']
    

def get_analyzers_results(strat: bt.Strategy) -> dict:
    """
    Extracts results from all analyzers attached to a strategy object.
    """
    results = {}
    for name, analyzer in strat.analyzers.getitems():
        results[name] = analyzer.get_analysis()

    return results
