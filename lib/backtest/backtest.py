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
    return results[0]


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
        self.sl_upper = self.p.spread_mean + self.p.spread_std * 3 * self.p.entry_factor
        self.sl_lower = self.p.spread_mean - self.p.spread_std * 3 * self.p.entry_factor

    def log(self, txt, dt=None):
        """ Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'[INFO ORDER] {dt.isoformat()} {txt}')

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
            if self.spread_indicator.spread[0] > self.upper_threshold and self.spread_indicator.spread[-1] > self.upper_threshold:
                self.sell(data=self.datas[0], size=size0)
                self.buy(data=self.datas[1], size=size1)
                self.position_opened = 'short_spread'
                self.log(f'OPEN (Short Spread): {size0} {self.datas[0]._name}, {size1} {self.datas[1]._name}')

            elif self.spread_indicator.spread[0] < self.lower_threshold and self.spread_indicator.spread[-1] < self.lower_threshold:
                self.sell(data=self.datas[1], size=size1)
                self.buy(data=self.datas[0], size=size0)
                self.position_opened = 'long_spread'
                self.log(f'OPEN (Long Spread): {size1} {self.datas[1]._name}, {size0} {self.datas[0]._name}')

        elif self.position_opened and not self.sl_triggered:  # Verify if a position is currently open
            mean_reverting = np.sign(self.spread_indicator.spread[0]) != np.sign(self.spread_indicator.spread[-1])
            stop_loss = self.spread_indicator.spread[0] > self.sl_upper or self.spread_indicator.spread[0] < self.sl_lower
            if mean_reverting:
                self.close(data=self.datas[0], size=abs(position0))
                self.close(data=self.datas[1], size=abs(position1))
                self.position_opened = False
                self.log(f'CLOSE (Mean Reversion): {position0} {self.datas[0]._name}, {position1} {self.datas[1]._name}')
            elif stop_loss:
                self.close(data=self.datas[0], size=abs(position0))
                self.close(data=self.datas[1], size=abs(position1))                    
                self.log(f'CLOSE (Stop Loss Reached): {position0} {self.datas[0]._name}, {position1} {self.datas[1]._name}')
                self.sl_triggered = True
                self.position_opened = 'sl_triggered'


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
    

def get_analyzers_results(strat: bt.Strategy) -> dict:
    """
    Extracts results from all analyzers attached to a strategy object.
    """
    results = {}
    for name, analyzer in strat.analyzers.getitems():
        results[name] = analyzer.get_analysis()

    return results
