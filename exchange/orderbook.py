import bisect
import heapq
import itertools
from collections import deque
from typing import Literal

from .models import Order, OrderResult, OrderStatus, Trade

class _SideBook:

    def __init__(self, is_bid):
        self.priceMap = {}
        self.sorted_prices = []
        self.is_bid = is_bid
        self.level_volume = {}

    def _remove_price_level(self, price):

        self.priceMap.pop(price, None)
        self.level_volume.pop(price, None)
        idx = bisect.bisect_left(self.sorted_prices, price)
        if idx < len(self.sorted_prices) and self.sorted_prices[idx] == price:
            del self.sorted_prices[idx]

    def add_order(self, order):
        price = order.price
        if price not in self.priceMap:
            self.priceMap[price] = deque()
            bisect.insort(self.sorted_prices, price)
            self.level_volume[price] = 0
        self.priceMap[price].append(order)
        self.level_volume[price] += order.remaining_quantity

    def remove_order(self, order):
        price = order.price
        q = self.priceMap.get(price)
        if q is None:
            return
        try:
            q.remove(order)
        except ValueError:
            return
        self.level_volume[price] = self.level_volume.get(price, 0) - order.remaining_quantity
        if self.level_volume.get(price, 0) <= 0:
            self._remove_price_level(price)

    def cancel_order(self, order):
        price = order.price
        vol = self.level_volume.get(price)
        if vol is None:
            return 
        self.level_volume[price] = vol - order.remaining_quantity
        if self.level_volume[price] <= 0:
            self._remove_price_level(price)

    def _update_fill(self, price, qty):
        self.level_volume[price] = self.level_volume.get(price, 0) - qty

    def _clean_level(self, price):
        if self.level_volume.get(price, 0) <= 0:
            self._remove_price_level(price)

    def best_price(self):
        if not self.sorted_prices:
            return None
        return self.sorted_prices[-1] if self.is_bid else self.sorted_prices[0]

    def best_queue(self):
        bp = self.best_price()
        if bp is None:
            return None
        return self.priceMap.get(bp)

    def volume_at_price(self, price):
        return self.level_volume.get(price, 0)

    def depth(self, levels):
        result = []
        if self.is_bid:
            prices = reversed(self.sorted_prices)
        else:
            prices = iter(self.sorted_prices)
        for p in prices:
            if len(result) >= levels:
                break
            vol = self.level_volume.get(p, 0)
            if vol > 0:
                result.append((p, vol))
        return result

    def is_empty(self):
        return len(self.sorted_prices) == 0


class _AssetBook:

    def __init__(self):
        self.bids = _SideBook(is_bid=True)
        self.asks = _SideBook(is_bid=False)


class OrderBook:

    def __init__(self):
        self._books = {}
        self._ordersByID = {}
        self._orderHistory = deque()
        self._trades = []
        self._timestamp = 0

    def _get_book(self, asset):
        book = self._books.get(asset)
        if book is None:
            book = _AssetBook()
            self._books[asset] = book
        return book

    def _next_ts(self):
        self._timestamp += 1
        return self._timestamp


    def _match(
        self,
        order,
        book,
    ):

        is_buy = order.side == "buy"
        contra_side = book.asks if is_buy else book.bids

        fills = []

        while order.remaining_quantity > 0 and not contra_side.is_empty():
            best_p = contra_side.best_price()
            if best_p is None:
                break

            if is_buy:
                if order.price is not None and order.price < best_p:
                    break
            else:
                if order.price is not None and order.price > best_p:
                    break

            queue = contra_side.priceMap[best_p]

            while order.remaining_quantity > 0 and queue:
                resting = queue[0]
                
                if resting.status == "cancelled" or resting.remaining_quantity <= 0:
                    queue.popleft()
                    continue

                fill_qty = min(order.remaining_quantity, resting.remaining_quantity)

                if is_buy:
                    trade = Trade(
                        asset=order.asset,
                        price=best_p,
                        quantity=fill_qty,
                        buy_order_id=order.order_id,
                        sell_order_id=resting.order_id,
                    )
                else:
                    trade = Trade(
                        asset=order.asset,
                        price=best_p,
                        quantity=fill_qty,
                        buy_order_id=resting.order_id,
                        sell_order_id=order.order_id,
                    )

                order.filled_quantity += fill_qty
                order.remaining_quantity -= fill_qty
                resting.filled_quantity += fill_qty
                resting.remaining_quantity -= fill_qty

                contra_side._update_fill(best_p, fill_qty)

                order.trades.append(trade)
                resting.trades.append(trade)
                fills.append(trade)
                self._trades.append(trade)

                if resting.remaining_quantity <= 0:
                    resting.status = "filled"
                    queue.popleft()
                else:
                    resting.status = "partial"

            contra_side._clean_level(best_p)

        return fills

    def _check_fok_feasibility(
        self,
        order,
        book,
    ):

        is_buy = order.side == "buy"
        contra_side = book.asks if is_buy else book.bids

        if contra_side.is_empty():
            return False

        needed = order.remaining_quantity

        if is_buy:
            price_iter = iter(contra_side.sorted_prices)
        else:
            price_iter = reversed(contra_side.sorted_prices)

        for i in price_iter:
            if needed <= 0:
                break
            if is_buy:
                if order.price is not None and order.price < i:
                    break
            else:
                if order.price is not None and order.price > i:
                    break

            queue = contra_side.priceMap.get(i)
            if queue is None:
                continue
            for i in queue:
                if i.status == "cancelled" or i.remaining_quantity <= 0:
                    continue
                needed -= i.remaining_quantity
                if needed <= 0:
                    break

        return needed <= 0
    def submit_order(
        self,
        order_id,
        order_owner,
        order_type: Literal["market", "limit", "fok", "ioc"],
        side: Literal["buy", "sell"],
        asset,
        quantity,
        price = None,
    ):

        order = Order(
            order_id=order_id,
            order_owner=order_owner,
            order_type=order_type,
            side=side,
            asset=asset,
            quantity=quantity,
            price=price,
            timestamp=self._next_ts(),
        )

        self._ordersByID[order_id] = order
        self._orderHistory.append(order)

        book = self._get_book(asset)

        if order_type == "fok":
            if not self._check_fok_feasibility(order, book):
                order.status = "cancelled"
                return OrderResult(
                    order_id=order_id,
                    status="cancelled",
                    fills=[],
                    remaining_quantity=order.remaining_quantity,
                )

        fills = self._match(order, book)

        if order.remaining_quantity <= 0:
            order.status = "filled"
        elif order_type == "limit":
            if order.filled_quantity > 0:
                order.status = "partial"
            own_side = book.bids if side == "buy" else book.asks
            own_side.add_order(order)
            if order.status == "pending":
                order.status = "pending"
        elif order_type in ("market", "ioc"):
            if order.filled_quantity > 0:
                order.status = "partial"
            else:
                order.status = "cancelled"
            order.remaining_quantity = 0 
        elif order_type == "fok":
            if order.remaining_quantity > 0:
                order.status = "cancelled"
            else:
                order.status = "filled"

        return OrderResult(
            order_id=order_id,
            status=order.status,
            fills=fills,
            remaining_quantity=order.remaining_quantity,
        )

    def cancel_order(self, order_id):

        order = self._ordersByID.get(order_id)
        if order is None:
            return False
        if order.status in ("filled", "cancelled"):
            return False
        if order.remaining_quantity <= 0:
            return False

        book = self._get_book(order.asset)
        own_side = book.bids if order.side == "buy" else book.asks
        own_side.cancel_order(order)

        order.status = "cancelled"
        return True

    def modify_order(
        self,
        order_id,
        new_quantity = None,
        new_price = None,
    ):

        order = self._ordersByID.get(order_id)
        if order is None:
            return False
        if order.status in ("filled", "cancelled"):
            return False
        if order.remaining_quantity <= 0:
            return False

        if new_quantity is not None:
            if new_quantity <= 0:
                return False
            if new_quantity <= order.filled_quantity:
                return False

        book = self._get_book(order.asset)
        own_side = book.bids if order.side == "buy" else book.asks

        loses_priority = False
        if new_price is not None and new_price != order.price:
            loses_priority = True
        if new_quantity is not None and new_quantity > order.quantity:
            loses_priority = True

        if loses_priority:
            own_side.remove_order(order)
            if new_price is not None:
                order.price = new_price
            if new_quantity is not None:
                old_remaining = order.remaining_quantity
                order.quantity = new_quantity
                order.remaining_quantity = new_quantity - order.filled_quantity
            order.timestamp = self._next_ts()
            own_side.add_order(order)
        else:
            if new_quantity is not None:
                old_remaining = order.remaining_quantity
                order.quantity = new_quantity
                order.remaining_quantity = new_quantity - order.filled_quantity
                delta = old_remaining - order.remaining_quantity
                if delta != 0:
                    price = order.price
                    own_side.level_volume[price] = (
                        own_side.level_volume.get(price, 0) - delta
                    )

        return True


    def get_best_bid(self, asset):
        book = self._books.get(asset)
        if book is None:
            return None
        bp = book.bids.best_price()
        if bp is None:
            return None
        vol = book.bids.volume_at_price(bp)
        if vol <= 0:
            return None
        return (bp, vol)

    def get_best_ask(self, asset):
        book = self._books.get(asset)
        if book is None:
            return None
        bp = book.asks.best_price()
        if bp is None:
            return None
        vol = book.asks.volume_at_price(bp)
        if vol <= 0:
            return None
        return (bp, vol)

    def get_spread(self, asset):
        bid = self.get_best_bid(asset)
        ask = self.get_best_ask(asset)
        if bid is None or ask is None:
            return None
        return ask[0] - bid[0]

    def get_market_depth(self, asset, levels = 5):
        book = self._books.get(asset)
        if book is None:
            return {"bids": [], "asks": []}
        return {
            "bids": book.bids.depth(levels),
            "asks": book.asks.depth(levels),
        }

    def get_volume_at_price(
        self, asset, price, side: Literal["bid", "ask"]
    ):
        book = self._books.get(asset)
        if book is None:
            return 0
        s = book.bids if side == "bid" else book.asks
        return s.volume_at_price(price)


    def view_orderbook(self):
        snapshot = {}
        for asset, book in self._books.items():
            bids_data = []
            for p in reversed(book.bids.sorted_prices):
                orders = [
                    o for o in book.bids.priceMap[p]
                    if o.status in ("pending", "partial") and o.remaining_quantity > 0
                ]
                if orders:
                    bids_data.append({"price": p, "orders": orders})
            asks_data = []
            for p in book.asks.sorted_prices:
                orders = [
                    o for o in book.asks.priceMap[p]
                    if o.status in ("pending", "partial") and o.remaining_quantity > 0
                ]
                if orders:
                    asks_data.append({"price": p, "orders": orders})
            snapshot[asset] = {"bids": bids_data, "asks": asks_data}
        return snapshot

    def view_last_orders(self, n = 10000):

        return list(itertools.islice(reversed(self._orderHistory), n))

    def view_largest_orders(self, n = 10000):
        eligible = [
            x for x in self._orderHistory if x.status != "cancelled"
        ]
        return heapq.nlargest(n, eligible, key=lambda o: o.quantity)

    def view_smallest_orders(self, n = 10000):
        eligible = [
            x for x in self._orderHistory if x.status != "cancelled"
        ]
        return heapq.nsmallest(n, eligible, key=lambda o: o.quantity)


    def get_trade_history(
        self,
        asset = None,
        limit = 1000,
    ):
        if asset is None:
            return list(reversed(self._trades))[:limit]
        return [
            x for x in reversed(self._trades) if x.asset == asset
        ][:limit]

    def get_order_status(self, order_id):
        order = self._ordersByID.get(order_id)
        if order is None:
            return None
        return OrderStatus(
            order_id=order.order_id,
            status=order.status,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.remaining_quantity,
            trades=list(order.trades),
        )
