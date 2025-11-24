import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from localpackage.utils import setup
from hyperliquid.utils import constants


def main():
    address, info, exchange = setup(constants.MAINNET_API_URL, skip_ws=True)

    print(address)
    # Filter dict 
    filtered_meta = [item for item in info.spot_meta()['tokens'] if item["name"] == "HYPE"][0]
    print(filtered_meta)
    print(info.user_state(address))
    print(info.spot_user_state(address))

if __name__ == "__main__":
    main()