"""
orderbook.py — High-performance limit order book & matching engine for CUATS.

=============================================================================
DATA STRUCTURE CHOICES & JUSTIFICATION
=============================================================================

1. Per-asset order book  (``_books: dict[str, _AssetBook]``)
   -------------------------------------------------------
   A top-level ``dict`` keyed by asset symbol gives O(1) dispatch to the
   relevant book for every operation.  Assets are created lazily on first
   use so the engine starts with zero allocation overhead.

2. Per-side price levels  (``price_map: dict[float, deque[Order]]``)
   ----------------------------------------------------------------
   Each side (bid / ask) of an ``_AssetBook`` stores its resting orders in

       price_map : dict[float, deque[Order]]

   *   ``dict``  → O(1) lookup of the queue at a given price.
   *   ``deque`` → O(1) appendleft / popleft → preserves FIFO (time
       priority) within a price level.  ``deque`` is implemented in C in
       CPython and is substantially faster than ``list`` for left-side
       pops.

3. Per-level volume tracker  (``_level_volume: dict[float, int]``)
   ---------------------------------------------------------------
   Each side tracks the *total resting quantity* at every active price
   level.  Updated in O(1) on add, cancel, fill, and modify.  This
   makes ``volume_at_price``, ``get_best_bid/ask``, ``get_spread``, and
   ``get_market_depth`` all O(1) or O(levels) without iterating orders.

4. Sorted price list  (``sorted_prices: list[float]``)
   ---------------------------------------------------
   A plain Python ``list`` kept sorted via ``bisect.insort``.

   *   Best price is always at a known index  →  O(1) peek.
       - Bids: ``sorted_prices[-1]``  (highest)
       - Asks: ``sorted_prices[0]``   (lowest)
   *   Insertion via ``bisect.insort``  →  O(log P) search + O(P) shift
       where P is the number of *distinct active price levels* on that
       side.  In practice P is far smaller than the total number of
       orders, so the shift cost is negligible.
   *   Removal of an exhausted price level  →  O(log P) search via
       ``bisect.bisect_left`` + O(P) worst-case shift.

   Alternatives considered and rejected:

   *   ``heapq`` — O(log P) insert, O(1) peek, but *no efficient
       arbitrary delete*.  Cancelling or exhausting a price level in the
       middle of the heap is O(P) with a rebuild or requires a lazy-
       deletion scheme that adds complexity and hurts cache locality.
   *   ``SortedList`` (sortedcontainers) — O(log P) insert *and* delete,
       but it is a third-party package.  The hackathon requires standard-
       library only (or self-implementation with justification).
   *   Red-black / AVL tree — not in the standard library; implementing
       one adds significant code for a marginal improvement over
       ``bisect`` when P is small.

   **Trade-off summary**: We accept O(P) worst-case shift on insert in
   exchange for O(1) best-price access, O(1) queue access, and
   simplicity.  Since the number of distinct price levels is typically
   orders of magnitude smaller than the number of orders, this is an
   excellent practical trade-off.

5. Lazy cancellation
   -----------------
   ``cancel_order`` does *not* scan the deque to physically remove the
   cancelled order (which would be O(Q)).  Instead it marks the order as
   cancelled and decrements the per-level volume counter in O(1).  The
   matching loop already skips cancelled orders at the front of the
   queue, popping them in O(1) each.  When a level's tracked volume
   reaches zero the entire level is removed.  This makes cancel O(1)
   amortised; stale entries are cleaned lazily during matching.

6. Order index  (``_orders_by_id: dict[int, Order]``)
   --------------------------------------------------
   O(1) lookup for cancel / modify / status queries.

7. Order history  (``_order_history: deque[Order]``)
   ------------------------------------------------
   Append-only, newest-first retrieval by ``itertools.islice`` on
   ``reversed()`` → O(n) for the requested *n*, not the full history.

8. Trade history  (``_trades: list[Trade]``)
   ----------------------------------------
   Append-only list; newest-first retrieval by reversed slicing.

=============================================================================
COMPLEXITY ANALYSIS  (P = price levels, Q = queue depth, K = fills,
                      N = total orders, n = requested count)
=============================================================================

  submit_order        →  O(log P + K)   bisect insert + K fills each O(1).
  cancel_order        →  O(1) amortised  mark + volume decrement.  Level
                          cleanup (when volume hits 0) is O(log P + P).
  modify_order        →  O(Q + log P)   eager deque removal + bisect
                          re-insert.
  get_best_bid/ask    →  O(1)           index into sorted_prices + volume
                          lookup from _level_volume.
  get_spread          →  O(1)           two best-price lookups.
  get_market_depth    →  O(levels)      iterate prices, read _level_volume.
  get_volume_at_price →  O(1)           dict lookup in _level_volume.
  view_last_orders    →  O(n)           islice on reversed deque.
  view_largest_orders →  O(N log n)     heapq.nlargest.
  view_smallest_orders→  O(N log n)     heapq.nsmallest.
  get_trade_history   →  O(T) or O(limit) reversed iteration.
  get_order_status    →  O(F)           O(1) lookup + O(F) fill copy.

=============================================================================
"""

from __future__ import annotations

import bisect
import heapq
import itertools
from collections import deque
from typing import Any, Dict, List, Literal, Optional, Tuple

from .models import Order, OrderResult, OrderStatus, Trade


# ========================================================================= #
# Internal helpers                                                           #
# ========================================================================= #

class _SideBook:
    """One side (bid or ask) of a single-asset order book.

    Maintains:
        price_map      : dict[float, deque[Order]]
        sorted_prices  : list[float]          — kept sorted ascending
        _level_volume  : dict[float, int]     — aggregate resting qty per level

    For bids the best price is ``sorted_prices[-1]`` (highest).
    For asks the best price is ``sorted_prices[0]``  (lowest).

    Complexity (P = number of distinct price levels, Q = queue depth):
        add_order      : O(log P)   (bisect.insort + volume update)
        remove_order   : O(Q)       (deque.remove for modify; eager)
        cancel_order   : O(1) amort (lazy mark + volume decrement)
        best_price     : O(1)       (index into sorted list)
        volume_at_price: O(1)       (dict lookup)
    """

    __slots__ = ("price_map", "sorted_prices", "is_bid", "_level_volume")

    def __init__(self, is_bid: bool) -> None:
        self.price_map: Dict[float, deque[Order]] = {}
        self.sorted_prices: List[float] = []
        self.is_bid = is_bid
        self._level_volume: Dict[float, int] = {}

    # ------------------------------------------------------------------ #
    # Price-level lifecycle                                                #
    # ------------------------------------------------------------------ #

    def _remove_price_level(self, price: float) -> None:
        """Remove all bookkeeping for *price*.  Idempotent.

        Uses ``bisect.bisect_left`` for O(log P) index search, then
        ``del sorted_prices[idx]`` for O(P) C-level shift.
        """
        self.price_map.pop(price, None)
        self._level_volume.pop(price, None)
        idx = bisect.bisect_left(self.sorted_prices, price)
        if idx < len(self.sorted_prices) and self.sorted_prices[idx] == price:
            del self.sorted_prices[idx]

    # ------------------------------------------------------------------ #
    # Mutations                                                            #
    # ------------------------------------------------------------------ #

    def add_order(self, order: Order) -> None:
        """Insert *order* into this side of the book.

        Complexity: O(log P) for bisect lookup + O(P) amortised shift.
        """
        price = order.price
        if price not in self.price_map:
            self.price_map[price] = deque()
            bisect.insort(self.sorted_prices, price)
            self._level_volume[price] = 0
        self.price_map[price].append(order)
        self._level_volume[price] += order.remaining_quantity

    def remove_order(self, order: Order) -> None:
        """Eagerly remove a specific *order* from its price level.

        Used by ``modify_order`` which needs the order physically removed
        before re-inserting at a new price/position.

        Complexity: O(Q) scan within the deque at this price + O(log P)
        bisect search if the level is cleaned up.
        """
        price = order.price
        q = self.price_map.get(price)
        if q is None:
            return
        try:
            q.remove(order)
        except ValueError:
            return
        self._level_volume[price] = self._level_volume.get(price, 0) - order.remaining_quantity
        if self._level_volume.get(price, 0) <= 0:
            self._remove_price_level(price)

    def cancel_order(self, order: Order) -> None:
        """Lazy-cancel: decrement volume without removing from the deque.

        The cancelled order remains in the deque and is skipped (and
        popped) lazily during matching.  When the level's tracked volume
        reaches zero the entire level is removed.

        Complexity: O(1) amortised.
        """
        price = order.price
        vol = self._level_volume.get(price)
        if vol is None:
            return  # level already cleaned up
        self._level_volume[price] = vol - order.remaining_quantity
        if self._level_volume[price] <= 0:
            self._remove_price_level(price)

    def _update_fill(self, price: float, qty: int) -> None:
        """Decrement level volume after a fill.  Called from ``_match``."""
        self._level_volume[price] = self._level_volume.get(price, 0) - qty

    def _clean_level(self, price: float) -> None:
        """Remove a price level if its tracked volume has been drained."""
        if self._level_volume.get(price, 0) <= 0:
            self._remove_price_level(price)

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def best_price(self) -> Optional[float]:
        """O(1) — best price on this side, or None."""
        if not self.sorted_prices:
            return None
        return self.sorted_prices[-1] if self.is_bid else self.sorted_prices[0]

    def best_queue(self) -> Optional[deque]:
        """O(1) — the deque at the best price level, or None."""
        bp = self.best_price()
        if bp is None:
            return None
        return self.price_map.get(bp)

    def volume_at_price(self, price: float) -> int:
        """Total resting volume at *price*.  O(1)."""
        return self._level_volume.get(price, 0)

    def depth(self, levels: int) -> List[Tuple[float, int]]:
        """Return up to *levels* ``(price, volume)`` pairs.

        Bids: descending price.  Asks: ascending price.
        Complexity: O(levels).
        """
        result: List[Tuple[float, int]] = []
        if self.is_bid:
            prices = reversed(self.sorted_prices)
        else:
            prices = iter(self.sorted_prices)
        for p in prices:
            if len(result) >= levels:
                break
            vol = self._level_volume.get(p, 0)
            if vol > 0:
                result.append((p, vol))
        return result

    def is_empty(self) -> bool:
        return len(self.sorted_prices) == 0


class _AssetBook:
    """Bid and ask sides for a single asset.

    Attributes
    ----------
    bids : _SideBook   (is_bid=True)
    asks : _SideBook   (is_bid=False)
    """

    __slots__ = ("bids", "asks")

    def __init__(self) -> None:
        self.bids = _SideBook(is_bid=True)
        self.asks = _SideBook(is_bid=False)


# ========================================================================= #
# Public OrderBook class                                                     #
# ========================================================================= #

class OrderBook:
    """High-performance limit order book and matching engine for CUATS.

    Instantiate with ``OrderBook()`` to create a fresh, empty exchange.
    All interaction happens through the public methods listed below — no
    other class needs to be instantiated by the caller.

    See module-level docstring for full data-structure rationale and
    complexity analysis.

    Internal state
    --------------
    _books         : dict[str, _AssetBook]
        Per-asset order books, created lazily.
    _orders_by_id  : dict[int, Order]
        Global index for O(1) cancel / modify / status.
    _order_history : deque[Order]
        Chronological record (newest first on retrieval).
    _trades        : list[Trade]
        Append-only trade tape.
    _timestamp     : int
        Monotonic counter for time priority.
    """

    def __init__(self) -> None:
        self._books: Dict[str, _AssetBook] = {}
        self._orders_by_id: Dict[int, Order] = {}
        self._order_history: deque[Order] = deque()
        self._trades: List[Trade] = []
        self._timestamp: int = 0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_book(self, asset: str) -> _AssetBook:
        """Return (or lazily create) the book for *asset*.  O(1)."""
        book = self._books.get(asset)
        if book is None:
            book = _AssetBook()
            self._books[asset] = book
        return book

    def _next_ts(self) -> int:
        self._timestamp += 1
        return self._timestamp

    # ------------------------------------------------------------------ #
    # Matching engine                                                      #
    # ------------------------------------------------------------------ #

    def _match(
        self,
        order: Order,
        book: _AssetBook,
    ) -> List[Trade]:
        """Core matching loop — price-time priority.

        Parameters
        ----------
        order : Order
            The incoming (aggressor) order.
        book : _AssetBook
            The asset's order book.

        Returns
        -------
        list[Trade]
            The fills produced.

        Complexity: O(K + S) where K is the number of individual fills
        and S is stale cancelled orders encountered and popped.  Each
        fill and each stale pop is O(1).  S is amortised across all
        cancel operations.
        """

        is_buy = order.side == "buy"
        contra_side: _SideBook = book.asks if is_buy else book.bids

        fills: List[Trade] = []

        while order.remaining_quantity > 0 and not contra_side.is_empty():
            best_p = contra_side.best_price()
            if best_p is None:
                break

            # Price crossing check
            if is_buy:
                if order.price is not None and order.price < best_p:
                    break
            else:
                if order.price is not None and order.price > best_p:
                    break

            queue = contra_side.price_map[best_p]

            while order.remaining_quantity > 0 and queue:
                resting = queue[0]

                # Skip cancelled orders that haven't been cleaned yet
                if resting.status == "cancelled" or resting.remaining_quantity <= 0:
                    queue.popleft()
                    continue

                fill_qty = min(order.remaining_quantity, resting.remaining_quantity)

                # Execute the fill
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

                # Update quantities
                order.filled_quantity += fill_qty
                order.remaining_quantity -= fill_qty
                resting.filled_quantity += fill_qty
                resting.remaining_quantity -= fill_qty

                # Update per-level volume tracker
                contra_side._update_fill(best_p, fill_qty)

                # Record trade on both orders
                order.trades.append(trade)
                resting.trades.append(trade)
                fills.append(trade)
                self._trades.append(trade)

                # Update resting order status
                if resting.remaining_quantity <= 0:
                    resting.status = "filled"
                    queue.popleft()
                else:
                    resting.status = "partial"

            # Clean up empty price level
            contra_side._clean_level(best_p)

        return fills

    def _check_fok_feasibility(
        self,
        order: Order,
        book: _AssetBook,
    ) -> bool:
        """Check whether a FOK order can be fully filled without mutating state.

        Walks the contra side's price levels in matching order and sums
        available volume at crossing prices.  No state is modified.

        Complexity: O(K) where K is the number of resting orders touched
        during the walk (same as actual matching would be).
        """
        is_buy = order.side == "buy"
        contra_side: _SideBook = book.asks if is_buy else book.bids

        if contra_side.is_empty():
            return False

        needed = order.remaining_quantity

        # Walk prices in matching order: asks ascending, bids descending
        if is_buy:
            price_iter = iter(contra_side.sorted_prices)
        else:
            price_iter = reversed(contra_side.sorted_prices)

        for price in price_iter:
            if needed <= 0:
                break
            # Price crossing check
            if is_buy:
                if order.price is not None and order.price < price:
                    break
            else:
                if order.price is not None and order.price > price:
                    break

            queue = contra_side.price_map.get(price)
            if queue is None:
                continue
            for resting in queue:
                if resting.status == "cancelled" or resting.remaining_quantity <= 0:
                    continue
                needed -= resting.remaining_quantity
                if needed <= 0:
                    break

        return needed <= 0

    # ------------------------------------------------------------------ #
    # Public API — Order Management                                        #
    # ------------------------------------------------------------------ #

    def submit_order(
        self,
        order_id: int,
        order_owner: str,
        order_type: Literal["market", "limit", "fok", "ioc"],
        side: Literal["buy", "sell"],
        asset: str,
        quantity: int,
        price: Optional[float] = None,
    ) -> OrderResult:
        """Submit an order to the exchange.

        The matching engine immediately attempts to fill the incoming order
        against the resting book (aggressor matching).  Unmatched residual
        for limit orders is placed on the book; market / IOC / FOK
        residuals are cancelled.

        Parameters
        ----------
        order_id   : unique order identifier
        order_owner: trader id
        order_type : ``"limit"`` | ``"market"`` | ``"fok"`` | ``"ioc"``
        side       : ``"buy"`` | ``"sell"``
        asset      : asset symbol
        quantity   : number of units (> 0)
        price      : required for ``"limit"``, ``"fok"``, ``"ioc"``; ignored
                     for ``"market"``

        Returns
        -------
        OrderResult

        Complexity: O(log P + K)
            log P  — potential insertion into sorted price list via bisect
            K      — number of fills during matching
        """

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

        self._orders_by_id[order_id] = order
        self._order_history.append(order)

        book = self._get_book(asset)

        # -------------------------------------------------------------- #
        # FOK: check feasibility before matching                          #
        # -------------------------------------------------------------- #
        if order_type == "fok":
            if not self._check_fok_feasibility(order, book):
                order.status = "cancelled"
                return OrderResult(
                    order_id=order_id,
                    status="cancelled",
                    fills=[],
                    remaining_quantity=order.remaining_quantity,
                )

        # -------------------------------------------------------------- #
        # Match                                                            #
        # -------------------------------------------------------------- #
        fills = self._match(order, book)

        # -------------------------------------------------------------- #
        # Post-match handling                                              #
        # -------------------------------------------------------------- #
        if order.remaining_quantity <= 0:
            order.status = "filled"
        elif order_type == "limit":
            # Rest the remaining quantity on the book
            if order.filled_quantity > 0:
                order.status = "partial"
            own_side = book.bids if side == "buy" else book.asks
            own_side.add_order(order)
            if order.status == "pending":
                order.status = "pending"
        elif order_type in ("market", "ioc"):
            # Cancel unfilled remainder
            if order.filled_quantity > 0:
                order.status = "partial"
            else:
                order.status = "cancelled"
            order.remaining_quantity = 0  # fully consumed / cancelled
        elif order_type == "fok":
            # Should be fully filled if we got here (feasibility passed)
            if order.remaining_quantity > 0:
                # Shouldn't happen, but safety net
                order.status = "cancelled"
            else:
                order.status = "filled"

        return OrderResult(
            order_id=order_id,
            status=order.status,
            fills=fills,
            remaining_quantity=order.remaining_quantity,
        )

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending (or partially filled) order.

        Uses lazy deletion: the order is marked cancelled and the per-
        level volume counter is decremented in O(1).  The order remains
        in the deque and is popped lazily during matching.

        Complexity: O(1) amortised.  Level cleanup (when volume hits 0)
        adds O(log P + P) but happens at most once per level creation.

        Returns
        -------
        bool — ``True`` if successfully cancelled.
        """
        order = self._orders_by_id.get(order_id)
        if order is None:
            return False
        if order.status in ("filled", "cancelled"):
            return False
        if order.remaining_quantity <= 0:
            return False

        # Lazy cancel: decrement volume, don't touch the deque
        book = self._get_book(order.asset)
        own_side = book.bids if order.side == "buy" else book.asks
        own_side.cancel_order(order)

        order.status = "cancelled"
        return True

    def modify_order(
        self,
        order_id: int,
        new_quantity: Optional[int] = None,
        new_price: Optional[float] = None,
    ) -> bool:
        """Modify an existing resting order.

        Rules (per specification):
        *   Price change → order loses time priority (re-queued).
        *   Quantity decrease → preserves time priority.
        *   Quantity increase → loses time priority (re-queued).
        *   New quantity must be > already-filled quantity.

        Complexity: O(Q + log P) — eager deque removal O(Q) + bisect
        re-insert O(log P).  Quantity-decrease-only is O(1).

        Returns
        -------
        bool — ``True`` if modification succeeded.
        """
        order = self._orders_by_id.get(order_id)
        if order is None:
            return False
        if order.status in ("filled", "cancelled"):
            return False
        if order.remaining_quantity <= 0:
            return False

        # Validate new_quantity
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
            # Remove, update, re-insert with new timestamp
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
            # Quantity decrease — keep position, update volume delta
            if new_quantity is not None:
                old_remaining = order.remaining_quantity
                order.quantity = new_quantity
                order.remaining_quantity = new_quantity - order.filled_quantity
                delta = old_remaining - order.remaining_quantity
                if delta != 0:
                    price = order.price
                    own_side._level_volume[price] = (
                        own_side._level_volume.get(price, 0) - delta
                    )

        return True

    # ------------------------------------------------------------------ #
    # Public API — Market Data                                             #
    # ------------------------------------------------------------------ #

    def get_best_bid(self, asset: str) -> Optional[Tuple[float, int]]:
        """Best (highest) bid price and total volume.

        Complexity: O(1).

        Returns
        -------
        tuple[float, int] | None
        """
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

    def get_best_ask(self, asset: str) -> Optional[Tuple[float, int]]:
        """Best (lowest) ask price and total volume.

        Complexity: O(1).

        Returns
        -------
        tuple[float, int] | None
        """
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

    def get_spread(self, asset: str) -> Optional[float]:
        """Bid-ask spread.  O(1).

        Returns
        -------
        float | None — ``None`` if either side is empty.
        """
        bid = self.get_best_bid(asset)
        ask = self.get_best_ask(asset)
        if bid is None or ask is None:
            return None
        return ask[0] - bid[0]

    def get_market_depth(self, asset: str, levels: int = 5) -> dict:
        """Market depth up to *levels* price levels per side.

        Returns
        -------
        dict with keys ``"bids"`` and ``"asks"``, each a list of
        ``(price, volume)`` tuples.

        Complexity: O(levels).
        """
        book = self._books.get(asset)
        if book is None:
            return {"bids": [], "asks": []}
        return {
            "bids": book.bids.depth(levels),
            "asks": book.asks.depth(levels),
        }

    def get_volume_at_price(
        self, asset: str, price: float, side: Literal["bid", "ask"]
    ) -> int:
        """Total resting volume at a specific price level.  O(1).

        Returns
        -------
        int — 0 if no orders at that level.
        """
        book = self._books.get(asset)
        if book is None:
            return 0
        s = book.bids if side == "bid" else book.asks
        return s.volume_at_price(price)

    # ------------------------------------------------------------------ #
    # Public API — Order Book Views                                        #
    # ------------------------------------------------------------------ #

    def view_orderbook(self) -> Any:
        """Return the current state of the limit order book for all assets.

        Returns a lightweight snapshot object with a ``@property`` for each
        asset, exposing bid and ask levels as read-only lists.

        Complexity: O(total resting orders) to build snapshot.
        """
        snapshot: Dict[str, dict] = {}
        for asset, book in self._books.items():
            bids_data = []
            for p in reversed(book.bids.sorted_prices):
                orders = [
                    o for o in book.bids.price_map[p]
                    if o.status in ("pending", "partial") and o.remaining_quantity > 0
                ]
                if orders:
                    bids_data.append({"price": p, "orders": orders})
            asks_data = []
            for p in book.asks.sorted_prices:
                orders = [
                    o for o in book.asks.price_map[p]
                    if o.status in ("pending", "partial") and o.remaining_quantity > 0
                ]
                if orders:
                    asks_data.append({"price": p, "orders": orders})
            snapshot[asset] = {"bids": bids_data, "asks": asks_data}
        return snapshot

    def view_last_orders(self, n: int = 10000) -> List[Order]:
        """Return the *n* most recent orders (newest first).

        Complexity: O(n).
        """
        return list(itertools.islice(reversed(self._order_history), n))

    def view_largest_orders(self, n: int = 10000) -> List[Order]:
        """Return the *n* largest orders by original quantity (non-cancelled).

        Sorted by quantity descending.

        Complexity: O(N log n) via ``heapq.nlargest``.
        """
        eligible = [
            o for o in self._order_history if o.status != "cancelled"
        ]
        return heapq.nlargest(n, eligible, key=lambda o: o.quantity)

    def view_smallest_orders(self, n: int = 10000) -> List[Order]:
        """Return the *n* smallest orders by original quantity (non-cancelled).

        Sorted by quantity ascending.

        Complexity: O(N log n) via ``heapq.nsmallest``.
        """
        eligible = [
            o for o in self._order_history if o.status != "cancelled"
        ]
        return heapq.nsmallest(n, eligible, key=lambda o: o.quantity)

    # ------------------------------------------------------------------ #
    # Public API — Trade History & Status                                  #
    # ------------------------------------------------------------------ #

    def get_trade_history(
        self,
        asset: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Trade]:
        """Return recent trades, optionally filtered by asset, newest first.

        Complexity: O(T) where T is total trades (or *limit*).
        """
        if asset is None:
            return list(reversed(self._trades))[:limit]
        return [
            t for t in reversed(self._trades) if t.asset == asset
        ][:limit]

    def get_order_status(self, order_id: int) -> Optional[OrderStatus]:
        """Snapshot of an order's current state.

        Complexity: O(1) lookup + O(F) to copy fills list.

        Returns
        -------
        OrderStatus | None
        """
        order = self._orders_by_id.get(order_id)
        if order is None:
            return None
        return OrderStatus(
            order_id=order.order_id,
            status=order.status,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.remaining_quantity,
            trades=list(order.trades),
        )
