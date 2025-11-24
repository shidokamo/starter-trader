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

    book = info.l2_snapshot(MARKET)['levels']
    print(book)

    print("BID:", book[0][0])
    print("ASK:", book[1][0])

if __name__ == "__main__":
    main()