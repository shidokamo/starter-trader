import argparse
from datetime import datetime, timedelta, timezone
# import urllib.parse
from time import sleep
from typing import Optional, Dict, Any, List
import os
import sys
import traceback
from functools import reduce
import pandas as pd
import numpy as np
import math


from localpackage.logger import logger
from localpackage.split import batched
from localpackage.rand import gen_random_orders

from localpackage.utils import setup
from hyperliquid.utils import constants
from hyperliquid.exchange import Exchange

def no_empty(string):
    if not string:
        raise argparse.ArgumentTypeError("Empty string was passed.")
    if string == 'None':
        raise argparse.ArgumentTypeError("None was passed.")
    return string

def run_perp(requests=None) -> None:
    try:
        # Get arguments from MongoDB
        parser = argparse.ArgumentParser(description="Ape bot")
        parser.add_argument("--quote", type=no_empty, choices=['USDC'], required=True)
        parser.add_argument("--base", type=no_empty, choices=['HYPE', 'ETH', 'BTC', 'SOL', 'ASTER', 'ZEC'], required=True)
        parser.add_argument("--freq", type=str, required=True)
        parser.add_argument("--quantile", type=float, required=True)
        parser.add_argument("--dip", type=float, required=True)
        parser.add_argument("--take_profit", type=float, required=True)
        parser.add_argument("--position_timeout_hours", type=float, required=True)
        parser.add_argument("--buy_partial_fills_timeout_hours", type=float, required=False)
        parser.add_argument("--window_hours", type=float, required=True)
        parser.add_argument("--split_orders", type=int, required=False, default=400)
        parser.add_argument("--sell_order_spread", type=float, required=False, default=0.001)
        parser.add_argument("--buy_order_spread", type=float, required=False, default=0.002)
        parser.add_argument("--leverage", type=float, required=False, default=1)

        # Parse arguments from env var
        arg_dict = {
            "quote":                  os.environ.get("QUOTE"),
            "base":                   os.environ.get("BASE"),
            "freq":                   os.environ.get("FREQ"),
            "quantile":               os.environ.get("QUANTILE"),
            "dip":                    os.environ.get("DIP"),
            "take_profit":            os.environ.get("TAKE_PROFIT"),
            "position_timeout_hours": os.environ.get("POSITION_TIMEOUT_HOURS"),
            "buy_partial_fills_timeout_hours": os.environ.get("BUY_PARTIAL_FILLS_TIMEOUT_HOURS"),
            "window_hours":           os.environ.get("WINDOW_HOURS"),
            "split_orders":           os.environ.get("SPLIT_ORDERS"),
            "sell_order_spread":      os.environ.get("SELL_ORDER_SPREAD"),
            "buy_order_spread":       os.environ.get("BUY_ORDER_SPREAD"),
            "leverage":               os.environ.get("LEVERAGE"),
        }
        argv = [f"--{k}={v}" for k,v in arg_dict.items()]
        args = parser.parse_args(argv)
        logger.info("--- args ---")
        logger.info(vars(args)) # Need to convert argparse Namespace object to dict for GCP log scanner

        address, info, exchange = setup(base_url=constants.MAINNET_API_URL, skip_ws=True)

        # Get token meta data (assuming that perp decimals are same as spot)
        perp_meta = info.meta()
        perp_market_meta = [item for item in perp_meta['universe'] if item["name"] == args.base][0]
        logger.info(perp_market_meta)
        size_decimals = perp_market_meta['szDecimals']
        max_leverage = perp_market_meta['maxLeverage']

        # TODO: if max_leverage is not matching with args.leverage, update max_leverage as pssible as we can.

        market_id = perp_market_meta['name']

        positions = info.user_state(address)
        logger.debug("Perp balance", positions) 
        logger.debug(positions)
        leverage = float(positions['marginSummary']['totalNtlPos']) / float(positions['marginSummary']['accountValue'])
        logger.info("Leverage : %f" % leverage)

        # def get_coin_balance(balances) -> List[dict]:
        #     return {coin['coin']:coin for coin in balances}
        # coins = get_coin_balance(balances)
        # logger.info(coins)

        # Get candles
        #candles = []
        
        # Get current timestamp from epoch in milliseconds
        current_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Start timestamp
        def get_start_timestamp(freq, window_size):
            match freq:
                case("1m"):
                    return current_timestamp - window_size * 60 * 1000
                case("5m"):
                    return current_timestamp - window_size * 5 * 60 * 1000
                case("15m"):
                    return current_timestamp - window_size * 15 * 60 * 1000
                case("30m"):
                    return current_timestamp - window_size * 30 * 60 * 1000
                case("1h"):
                    return current_timestamp - window_size * 60 * 60 * 1000
                case("4h"):
                    return current_timestamp - window_size * 4 * 60 * 60 * 1000

        # TODO: use duplicated()?
        def unique(seq):
            seen = []
            unique_list = [x for x in seq if x not in seen and not seen.append(x)]
            return len(seq) == len(unique_list)
                
        # Candle
        candles = info.candles_snapshot(
            market_id,
            args.freq,
            get_start_timestamp(args.freq, 2000), # 2000 is the maximum number of candles we can get (I haven't tested this yet)
            current_timestamp
        )

        df = pd.DataFrame(candles)
        # Rename columns
        df = df.rename(columns={'t': "open-time", 'T': "close-time", "s": "market", "i": "frequency", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "count"})
        df = df.astype({'open-time': np.int64, 'open':np.float32, 'high':np.float32, 'low':np.float32, 'close':np.float32})
        df['open-time'] =  pd.to_datetime(df['open-time'], utc=True, unit='ms')
        df['close-time'] =  pd.to_datetime(df['close-time'], utc=True, unit='ms')

        df.set_index('open-time', inplace=True)
        if len(df[df.index.duplicated()]) > 0:
            raise Exception("There were duplicated index on candles")

        logger.info("Candle length : %d" % len(candles))
        logger.info("Candle uniqueness : %s " % unique(candles))
        logger.info("Candle start time : %s" % df.index[0])
        logger.info("Candle end time : %s" % df.index[-1])

        # error if low range hours is larger than candle range
        if args.window_hours > (df.index[-1] - df.index[0]).total_seconds() / 3600:
            logger.error("Candle low range hours is larger than candle range. Please check your candle freq.")
            return 'ERROR'

        # Get quantile low price for specified date range as hours
        end_time = df.index[-1]
        start_time = end_time - pd.Timedelta(hours=args.window_hours)
        mask = (df.index >= start_time)
        quantile_value = df.loc[mask, 'high'].quantile(args.quantile)
        # Get the actual value from the dataframe that matches the quantile value (or closest)
        high = df.loc[mask, 'high'].iloc[(df.loc[mask, 'high'] - quantile_value).abs().argmin()]
        high_id = df[df['high']==high].index[-1]
        logger.info("Quantile high price df start time : %s" % start_time)
        logger.info("Quantile high price for last %f hours : %f @ %s (%d candles ago, %f hours ago)" %
                     (args.window_hours, high, high_id, df.index.get_loc(high_id)-len(df), (end_time - high_id).total_seconds() / 3600))

        dip_price = high * args.dip
        #print(df)

        # Bid/Ask
        book = info.l2_snapshot(market_id)['levels']
        ask_best = book[1][0]
        bid_best = book[0][0]
        logger.info("Best ask : %s" % ask_best)
        logger.info("Best bid : %s" % bid_best)
        ask_price = float(ask_best['px'])
        bid_price = float(bid_best['px'])

        logger.info("Quantile high price : %f @ %s (%d candles)" % (high, high_id, df.index.get_loc(high_id)-len(df)))
        logger.info("Bid price : %f (%.2f%% from high price)" % (bid_price, (bid_price - high) / high * 100))
        logger.info("Dip price : %f (%.2f %% from bid price, %.2f %% from high price)" % (
            dip_price,
            (dip_price - bid_price) / bid_price * 100,
            (dip_price - high) / high * 100
        ))

        # Current orders
        orders = info.open_orders(address)
        orders = [x for x in orders if x['coin'] == market_id]
        # logger.debug(orders)

        def show_orders(orders):
            # logger.debug(orders)
            for x in orders:
                logger.debug("%s Price: %.5g, Size: %f (USD: %f)" % (x['side'], float(x['limitPx']), float(x['sz']), float(x['limitPx'])*float(x['sz'])))  # TODO: use size decimals

        # def take_matchings(arr, key, value):
        #     i = next((i for i, x in enumerate(arr) if x[key] != value), len(arr))
        #     return arr[:i] if i else []
        def take_matchings(arr, key, value):
            result = []
            for item in arr:
                if item[key] == value:
                    result.append(item)
                else:
                    break
            return result

        # Buy orders
        if os.environ.get("NO_BUY"):
            logger.warning("No buy option is enabled. Cancel all buy order and skip buy orders.")
            cancel = [{'coin':x['coin'], 'oid':x['oid']} for x in orders if x['side'] == 'B' ]
            exchange.bulk_cancel(cancel)
        else:
            # Get last buy order price
            fills = info.user_fills(address)
            fills = [x for x in fills if x['coin'] == market_id]

            last_buy_fills = take_matchings(fills, 'side', 'B')
            last_sell_fills = take_matchings(fills, 'side', 'A')
            if last_buy_fills:
                total_last_buy_fills = reduce(lambda sum, x: sum + float(x['sz']) * float(x['px']), last_buy_fills, 0)
            else:
                total_last_buy_fills = 0
            if last_sell_fills:
                total_last_sell_fills = reduce(lambda sum, x: sum + float(x['sz']) * float(x['px']), last_sell_fills, 0)
            else:
                total_last_sell_fills = 0
                
            logger.info("Total last buy fills: %f" % total_last_buy_fills)
            logger.info("Total last sell fills: %f" % total_last_sell_fills)

            ordered_sell = [x for x in fills if x['side'] == 'A' and x['coin'] == market_id]
            # Check last filled time
            ordered_sell_sorted = sorted(ordered_sell, key=lambda x: x['time'])
            # logger.debug(ordered_sell_sorted)
            if ordered_sell_sorted:
                # NOTE: You can implement cool time using below info
                last_filled_sell_order = ordered_sell_sorted[-1]
                logger.debug(last_filled_sell_order)
                last_filled_sell_time = datetime.fromtimestamp(float(last_filled_sell_order['time'])/1000.0, timezone.utc)
                last_filled_sell_price = float(last_filled_sell_order['px'])
                logger.info("Last filled sell order : %s, Time : %s, Price : %s, Size : %s"  % (last_filled_sell_order['side'], last_filled_sell_time, last_filled_sell_order['px'], last_filled_sell_order['sz']))

            orders_buy = [x for x in orders if x['side'] == 'B']
            if orders_buy:
                total_order_coin_size = reduce(lambda sum, x: sum + float(x['sz']) if x['side'] == 'B' else sum, orders, 0)
                total_order_size = reduce(lambda sum, x: sum + float(x['limitPx'])*float(x['sz']) if x['side'] == 'B' else sum, orders, 0)
                logger.info("Total open buy order USD size: %f" % total_order_size)
                logger.info("Total open buy order coin size: %f" % total_order_coin_size)
                if total_order_coin_size != 0:
                    total_order_price = total_order_size / total_order_coin_size
                    logger.info("Total buy order price: %f" % total_order_price)
                    min_buy = min([float(x['limitPx']) for x in orders_buy])
                    logger.info("Min buy order price: %f" % min_buy)
                    max_buy = max([float(x['limitPx']) for x in orders_buy])
                    logger.info("Max buy order price: %f" % max_buy)
                    count_buy = len(orders_buy)
                    logger.info("Buy order count: %d" % count_buy)
                else:
                    total_order_price = None
            else:
                total_order_coin_size = 0
                total_order_size = 0

            logger.info("Total account value : %f" % float(positions['marginSummary']['accountValue']))
            total_order_size_target = float(positions['marginSummary']['accountValue']) * args.leverage
            logger.info("Target total buy order USD size: %f" % total_order_size_target)

            # price calculation
            buy_spread = dip_price*args.buy_order_spread
            price = float("%.5g"%dip_price)  # This can cause immediate market order if dip_price is higher than best bid.
            logger.info("New Order Price : %.5g" % price)
            logger.info("New Order USD size : %f" % total_order_size_target) 

            # TODO: Prevent the case where all usd is used for margin.
            if leverage > args.leverage:
                logger.warning("Leverage is higher than target leverage. Skip buy orders. Cancel all buy orders.")
                cancel = [{'coin':x['coin'], 'oid':x['oid']} for x in orders if x['side'] == 'B' ]
                exchange.bulk_cancel(cancel)
            elif total_order_size_target < 10: # Minimum order size is 10 USD
                logger.warning("Account value is too low. Skip buy orders.")
            elif total_last_buy_fills > 0:  # Skip buy orders until sell order is filled
                # We only needs to do this when there is open buy orders
                if orders and total_order_coin_size > 0:
                    first_buy_fill = last_buy_fills[-1]
                    first_buy_fill_time = datetime.fromtimestamp(float(first_buy_fill['time'])/1000.0, timezone.utc)
                    logger.info("First partially filled buy order : %s, Time : %s, Price : %s, Size : %s"  % (first_buy_fill['side'], first_buy_fill_time, first_buy_fill['px'], first_buy_fill['sz']))
                    if datetime.now(timezone.utc) - first_buy_fill_time > timedelta(hours=args.buy_partial_fills_timeout_hours):
                        logger.warning("Buy order partial fills are too old. Cancel all remaining buy orders.")
                        cancel = [{'coin':x['coin'], 'oid':x['oid']} for x in orders if x['side'] == 'B' ]
                        exchange.bulk_cancel(cancel)
                    else:
                        logger.warning("Buy order partial fills are detected. Skip new buy order until sell order is filled.")
                        logger.info("Remaining partial buy order time : %s" % (timedelta(hours=args.buy_partial_fills_timeout_hours) - (datetime.now(timezone.utc) - first_buy_fill_time)))
                else:
                    logger.warning("Last buy fills are detected. Skip new buy order until sell order is filled.")
            elif (
                    not orders
                    or not total_order_price
                    or total_order_price / price < 0.999 # 0.1 % tolerance
                    or total_order_price / price > 1.001
                    or total_order_size / total_order_size_target < 0.999 # Order size also needs to be checked for order size mismatch with partial fill
                    or total_order_size / total_order_size_target > 1.001
                ):
                #logger.warning("Existing orders price doesn't match with current dip_price or there are no orders. Update orders.")
                cancel = [{'coin':x['coin'], 'oid':x['oid']} for x in orders if x['side'] == 'B' ]
                exchange.bulk_cancel(cancel)
                size = float(f"%.{size_decimals}f"%(total_order_size_target / price))  # This is not necessary for hyperliquid sdk. It can accept float size beyond decimals.
                logger.info("New Order Coin Size : %f" % size)
                split_orders = max(1, min(args.split_orders, int(size*price / 20))) # Actual min order size is 10 but we use 20 as min size
                logger.info("Split orders : %d" % split_orders)

                # Get random split orders
                if split_orders > 1: # Minimum order size is 10 USD
                    size_price = gen_random_orders(
                        n=split_orders,
                        total_coin_size=size,
                        center_price=price,
                        price_half_range=buy_spread,
                        size_decimals=size_decimals
                    )
                    # logger.debug(size_price)
                    buy_orders = [{
                        "coin": market_id,
                        "is_buy": True,
                        "sz": float(x['size']),
                        "limit_px": float(x['price']),
                        "order_type": {"limit": {"tif": "Gtc"}},
                        "reduce_only": False } for x in size_price]
                    # logger.debug(buy_orders)
                    order_result = exchange.bulk_orders(buy_orders)
                    logger.debug(order_result)
                else:
                    order_result = exchange.order(market_id, True, size, price, {"limit": {"tif": "Gtc"}})
                    logger.debug(order_result)
                # Check orders
                logger.debug("Check new orders")
                new_orders = info.open_orders(address)
                show_orders([x for x in new_orders if x['coin'] == market_id and x['side'] == 'B'])
                sleep(3) # Delay for order execution before sell order block
            else:
                logger.info("Existing buy orders are matching with current dip_price. Skip new buy order.")

        # Get position again to make sure we can check the filled positions
        positions = info.user_state(address)
        logger.debug("Perp balance")
        logger.debug(positions)

        position = [x['position'] for x in positions['assetPositions'] if x['position']['coin'] == market_id]
        # Like this
        # {'coin': 'HYPE', 'px': '32.703', 'sz': '4.94', 'side': 'B', 'time': 1734844131234, 'startPosition': '20067.16', 'dir': 'Open Long', 'closedPnl': '0.0', 'hash': '0xd9825f6dc4428cf9d01d0419c9105a0201e0008793e4437be9beb31a637b4869', 'oid': 57626339003, 'crossed': False, 'fee': '0.0', 'tid': 312077436344773, 'feeToken': 'USDC'}
        if position:
            logger.debug(position)
        
        # Sell orders
        if os.environ.get("NO_SELL"):
            logger.warning("No sell option is enabled. Cancel all sell order and skip sell orders.")
            cancel = [{'coin':x['coin'], 'oid':x['oid']} for x in orders if x['side'] == 'A' ]
            exchange.bulk_cancel(cancel)
        elif ( position and float(position[0]['szi']) > 0 ):
            position = position[0]
            entry_price = float(position['entryPx'])
            logger.info("Position size : %s" % position['szi'])
            logger.info("Entry price : %f" % entry_price)
            logger.info("Unrealized profit : %s" % position['unrealizedPnl'])
            logger.info("Liquidation price : %s" % position['liquidationPx'])

            # !!! Need to get fresh fills to make sure that we are selling coins that we just bouht in extreme case.
            fills = info.user_fills(address)
            fills = [x for x in fills if x['coin'] == market_id]

            last_buy_fills = take_matchings(fills, 'side', 'B')
            last_sell_fills = take_matchings(fills, 'side', 'A')
            if last_buy_fills:
                total_last_buy_fills = reduce(lambda sum, x: sum + float(x['sz']) * float(x['px']), last_buy_fills, 0)
            else:
                total_last_buy_fills = 0
            if last_sell_fills:
                total_last_sell_fills = reduce(lambda sum, x: sum + float(x['sz']) * float(x['px']), last_sell_fills, 0)
            else:
                total_last_sell_fills = 0
                
            logger.info("Total last buy fills: %f" % total_last_buy_fills)
            logger.info("Total last sell fills: %f" % total_last_sell_fills)

            ordered_buy = [x for x in fills if x['side'] == 'B' and x['coin'] == market_id]
            # Check last filled time
            ordered_buy_sorted = sorted(ordered_buy, key=lambda x: x['time'])
            # logger.debug(ordered_buy_sorted)
            last_filled_order = ordered_buy_sorted[-1]
            # logger.debug(last_filled_order)
            last_buy_filled_time = datetime.fromtimestamp(float(last_filled_order['time'])/1000.0, timezone.utc)
            logger.info("Last filled order : %s, Time : %s, Price : %s, Size : %s"  % (last_filled_order['side'], last_buy_filled_time, last_filled_order['px'], last_filled_order['sz']))

            # Get target sell prices
            take_profit_limit = float(entry_price) * args.take_profit
            logger.info("take_profit_limit : %f " % (take_profit_limit))
            if bid_price != 0:
                logger.info("%f +%f (%f%%) to take profit)" % (bid_price, take_profit_limit-bid_price, (take_profit_limit-bid_price)/bid_price*100))

            timeout = False
            if datetime.now(timezone.utc) - last_buy_filled_time > timedelta(hours=args.position_timeout_hours):
                logger.warning("Position timeout!! Just sell all on best asks")
                take_profit_limit = float(ask_price)
                timeout = True
            else:
                logger.info("Remaining position time : %s" % (timedelta(hours=args.position_timeout_hours) - (datetime.now(timezone.utc) - last_buy_filled_time)))
            logger.info("Target sell order price : %f" % take_profit_limit)

            # Open order
            orders_sell = [x for x in orders if x['side'] == 'A']
            if orders_sell:
                total_order_coin_size = reduce(lambda sum, x: sum + float(x['sz']) if x['side'] == 'A' else sum, orders, 0)
                total_order_size = reduce(lambda sum, x: sum + float(x['limitPx'])*float(x['sz']) if x['side'] == 'A' else sum, orders, 0)
                logger.info("Total open sell order USD size: %f" % total_order_size)
                logger.info("Total open sell order coin size: %f" % total_order_coin_size)
                if total_order_coin_size != 0:
                    total_order_price = total_order_size / total_order_coin_size
                    logger.info("Total sell order price: %f" % total_order_price)
                    min_sell = min([float(x['limitPx']) for x in orders_sell])
                    logger.info("Min sell order price: %f" % min_sell)
                    max_sell = max([float(x['limitPx']) for x in orders_sell])
                    logger.info("Max sell order price: %f" % max_sell)
                    count_sell = len(orders_sell)
                    logger.info("Sell order count: %d" % count_sell)

            #order_coin_size = math.floor(float(position['szi'])) # To pervent insufficient balance error in any case, make order size slithly smaller than position size.
            order_coin_size =float(position['szi'])
        
            # price calculation
            sell_spread = take_profit_limit*args.sell_order_spread
            price = float("%.5g" % take_profit_limit) # Note: If we send order with higher price than best bid, it will be executed
            
            total_order_size_target = order_coin_size * price
            logger.info("Target total sell order coin size: %f" % order_coin_size)
            logger.info("Target total sell order USD size: %f" % total_order_size_target)
            logger.info("Target sell order profit USD: %f" % (total_order_size_target - total_last_buy_fills))

            if not order_coin_size > 0:
                logger.warning("Position size is too low. Skip sell orders.")
            elif total_last_sell_fills > 0 and not timeout:
                logger.warning("Last sell fills are detected before position timed out. Skip new sell order until position times out.")
            elif (
                not orders_sell
                # Need to check usd order size so that we can update price if sell margin is changed
                or total_order_size / total_order_size_target < 0.995  # 0.5% tolerance
                or total_order_size / total_order_size_target > 1.005
            ):
                logger.warning("Existing sell orders doesn't match with take_profit_limit config. Update sell orders.")
                cancel = [{'coin':x['coin'], 'oid':x['oid']} for x in orders if x['side'] == 'A' ]
                exchange.bulk_cancel(cancel)

                logger.info("Sell Price : %.5g" % price)
                # size = float(f"%.{base_decimals}f"%(total_order_size_target / price))
                size = float(f"%.{size_decimals}f"%(order_coin_size)) # This is not necessary for hyperliquid sdk.
                logger.info("Sell Size : %f" % size)

                split_orders = max(1, min(args.split_orders, int(size*price / 100))) # Actual min order size is 10 but we use 100 as min size
                logger.info("Split orders : %d" % split_orders)

                if split_orders > 1:
                    size_price = gen_random_orders(
                        n=split_orders,
                        total_coin_size=size,
                        center_price=price,
                        price_half_range=sell_spread,
                        size_decimals=size_decimals,
                    )
                    logger.debug(size_price)
                    sell_orders = [{
                        "coin": market_id,
                        "is_buy": False,
                        "sz": float(x['size']),
                        "limit_px": float(x['price']),
                        "order_type": {"limit": {"tif": "Gtc"}},
                        "reduce_only": True } for x in size_price]
                    logger.debug(sell_orders)
                    order_result = exchange.bulk_orders(sell_orders)
                    logger.debug(order_result)                    
                else:
                    order_result = exchange.order(market_id, False, size, price, {"limit": {"tif": "Gtc"}})
                    logger.debug(order_result)
                # Check orders
                logger.debug("Check new orders")
                new_orders = info.open_orders(address)
                show_orders([x for x in new_orders if x['coin'] == market_id and x['side'] == 'A'])

        return 'OK' # HTTP request return value should be specifi
    except Exception as e:
        logger.exception("Exception in run command.")
        exc_info = sys.exc_info()
        logger.exception(traceback.format_exception(*exc_info))
        return 'ERROR' # HTTP request return value should be specifi


if __name__ == "__main__":
    run_perp()

