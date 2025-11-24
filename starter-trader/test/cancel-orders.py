import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from localpackage.utils import setup
from hyperliquid.utils import constants


def main():
    address, info, exchange = setup(constants.MAINNET_API_URL, skip_ws=True)

    open_orders = info.open_orders(address)

    print(open_orders)

    for open_order in open_orders:
        print(f"cancelling order {open_order}")
        exchange.cancel(open_order["coin"], open_order["oid"])


if __name__ == "__main__":
    main()