import random
import math

MIN_SIZE = 10
MAX_DECIMALS = 5 # Hyperliquid spec

def gen_random_with_fixed_sum(half_n, half_total):
    rand_n = [ random.random() for i in range(half_n) ]
    result = [ math.floor(i * half_total / sum(rand_n)) for i in rand_n]
    # drop result if it is less than MIN_SIZE

    for i in range(half_total - sum(result)):
        result[random.randint(0,half_n-1)] += 1
    result += reversed(result)
    return result

def gen_prices(half_n, center_price, half_range):
    price_offsets = [random.randint(0, half_range) for i in range(half_n)]
    prices = list(map(lambda x: center_price + x, price_offsets))
    prices += list(map(lambda x: center_price - x, price_offsets))
    return sorted(prices)

def gen_random_orders(n, total_coin_size, center_price, price_half_range, size_decimals):
    # odd number is always converted to even number
    half_n = int(n/2)
    normalized_center_price = int(center_price * (10 ** MAX_DECIMALS))
    normalized_price_half_range = int(price_half_range * (10 ** MAX_DECIMALS))
    print(f"Generating {n} orders with center price {normalized_center_price/(10 ** MAX_DECIMALS)} and half range {normalized_price_half_range/(10 ** MAX_DECIMALS)}")
    normalized_prices = gen_prices(half_n, normalized_center_price, normalized_price_half_range)
    # print(normalized_prices)

    # Get sizes for half of the orders
    normalized_size = int(total_coin_size * (10 ** size_decimals))
    #print("normalized size: %d" % normalized_size)
    #print("half n: %d" % half_n)
    normalized_sizes = gen_random_with_fixed_sum(half_n, int(normalized_size/2))
    #print(normalized_sizes)

    def fmt(x):
        return f"{x:.5g}" # 有効数字5桁

    return [{
        "price": fmt(normalized_prices[i]/(10 ** MAX_DECIMALS)),
        "size": (f"%.{size_decimals}f") % (normalized_sizes[i]/(10 ** size_decimals))
        } for i in range(half_n*2) if normalized_sizes[i]/(10 ** size_decimals)*(normalized_prices[i]/(10 ** MAX_DECIMALS)) >= MIN_SIZE]
