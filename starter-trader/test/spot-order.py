import json

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from localpackage.utils import setup

from hyperliquid.utils import constants

PURR = "HYPE/USDC"

def main():
    address, info, exchange = setup(base_url=constants.MAINNET_API_URL, skip_ws=True)

    # Get the user state and print out position information
    spot_user_state = info.spot_user_state(address)
    if len(spot_user_state["balances"]) > 0:
        print("spot balances:")
        for balance in spot_user_state["balances"]:
            print(json.dumps(balance, indent=2))
    else:
        print("no available token balances")

    # Place an order that should rest by setting the price very low
    order_result = exchange.order(PURR, True, 100, 5, {"limit": {"tif": "Gtc"}})
    print(order_result)

    # Query the order status by oid
    if order_result["status"] == "ok":
        status = order_result["response"]["data"]["statuses"][0]
        if "resting" in status:
            order_status = info.query_order_by_oid(address, status["resting"]["oid"])
            print("Order status by oid:", order_status)

    # Cancel the order
    if order_result["status"] == "ok":
        status = order_result["response"]["data"]["statuses"][0]
        if "resting" in status:
            cancel_result = exchange.cancel(PURR, status["resting"]["oid"])
            print(cancel_result)

if __name__ == "__main__":
    main()