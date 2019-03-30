import os
from datetime import datetime
import numpy as np
import logging
import pandas as pd
import time

# Sliding window params
window_size = 600
complexity_steps = 10

# General settings
filename = '^GSPC_1950.csv'
csv_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), filename)
news_csv_path = os.path.join(os.path.dirname(__file__), "NLP", "data", "nlp_news.csv")
feature_reuters_csv_path = os.path.join(os.path.dirname(__file__), "NLP", "data", "nlp_reuters_features.csv")
logger = logging.getLogger()
pkl_path = os.path.join(os.path.dirname(__file__), "Pickled ten year filtered data (Articles + DJIA).pkl")

threshold = 0.28  # defined threshold regarding classification 'Same'
classes = ["Up", "Same", "Down"]  # the classifications

# MA params
ma_fast = 13
ma_slow = 26
ma_50 = 50
ma_200 = 200
ma = 20
ws = 5  # for MA

alpha = 0.6 # used for evaluation mode 'weighted'
