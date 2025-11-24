import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from localpackage.utils import setup
from hyperliquid.utils import constants


def main():
    address, info, exchange = setup(constants.MAINNET_API_URL, skip_ws=True)

    fills = info.user_fills(address)

    print(fills)

    for fill in fills:
        print(f"Filled: {fill}")
        # exchange.cancel(open_order["coin"], open_order["oid"])


if __name__ == "__main__":
    main()