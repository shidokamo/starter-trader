import datetime
import json

import sys
import os
import pandas as pd
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from localpackage.utils import setup

from hyperliquid.utils import constants

MARKET = "HYPE/USDC"

def main():
    address, info, exchange = setup(base_url=constants.MAINNET_API_URL, skip_ws=True)

    # Get current timestamp from epoch in milliseconds
    current_timestamp = int(datetime.datetime.now().timestamp() * 1000)

    # 200 minutes ago
    start_timestamp = current_timestamp - 200 * 60 * 1000
    
    # Get the user state and print out position information
    candles = info.candles_snapshot(MARKET, '1m', start_timestamp, current_timestamp)

    df = pd.DataFrame(candles)
    # Rename columns
    df = df.rename(columns={'t': "open-time", 'T': "close-time", "s": "market", "i": "frequency", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "count"})
    df = df.astype({'open-time': np.int64, 'open':np.float32, 'high':np.float32, 'low':np.float32, 'close':np.float32})
    df['open-time'] =  pd.to_datetime(df['open-time'], utc=True, unit='ms')
    df['close-time'] =  pd.to_datetime(df['close-time'], utc=True, unit='ms')

    df.set_index('open-time', inplace=True)
    if len(df[df.index.duplicated()]) > 0:
        raise Exception("There were duplicated index")
    print(df)
    

if __name__ == "__main__":
    main()