import threading 
import numpy as np 
import matplotlib.pyplot as plt 
plt.switch_background('agg')

from mplfinance.original_flavor import candlestick_ohlc
from agent import Agent

lock = threading.Lock()


class Visualizer:
    COLORS = ['r', 'b', 'g']

    def __init__(self, vnet=False):
        self.canvas = None 
        self.fig = None 
        self.axes = None 
        self.title = ''
    
    def prepare(self, chart_data, title):
        self.title = title 