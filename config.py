import os
from datetime import datetime

csv_path = os.path.join(os.path.dirname(__file__), '^GSPC_1950.csv')
ma = 20
ws = 5
threshold = 0.1
classes = ["Up", "Same", "Down"]
# classes = ["Up", "Down"]
# threshold = 0

ma_fast = 13
ma_slow = 26
ma_50 = 50
ma_200 = 200

