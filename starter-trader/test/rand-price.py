import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from functools import reduce
from localpackage.rand import gen_random_orders

coin_size = 5003.12
price = 13.52
price_range = 0.05
orders = gen_random_orders(20, coin_size, price, price_range, size_decimals=2)

print("total order size", coin_size)
print("Dip price", price)
for x in orders:
    print(x)

# Total size
total_coin_size = reduce(lambda sum, x: sum + float(x['size']), orders, 0)
total_size = reduce(lambda sum, x: sum + float(x['price'])*float(x['size']) , orders, 0)
print("Total coin size: %f" % total_coin_size)
print("Total order size: %f" % total_size)
total_order_price = total_size / total_coin_size
print("Total buy order price: %f" % total_order_price)

