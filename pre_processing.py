import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
import fix_yahoo_finance as yf
yf.pdr_override()


symbols = ['^GSPC']
start_date = '2000-01-01'
end_date = '2018-07-27'

for symbol in symbols:
    data.get_data_yahoo(symbol, start_date, end_date).to_csv(symbol + '.csv')

