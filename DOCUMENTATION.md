# CUATS Hackathon 2026 — Orderbook Documentation

## 1. Data Structures

### Per-asset dispatch — `dict[str, _AssetBook]`
Routes operations to the correct book in O(1). Books created lazily on first use. A `list[_AssetBook]` would require an O(N) linear scan over asset names to find the right book.

### Price levels — `dict[float, deque[Order]]`
Maps each price to a FIFO queue. `dict` gives O(1) price lookup; `deque` gives O(1) append/popleft for time-priority matching (`list.pop(0)` would be O(n)).

### Volume tracker — `dict[float, int]`
Tracks total resting volume per price level, updated in O(1) on every add, cancel, fill, and modify. Eliminates deque iteration for volume queries — `get_best_bid`, `get_best_ask`, `get_volume_at_price` are all O(1).

### Sorted price index — `list[float]` + `bisect`
Sorted list of active prices. Best price is O(1): `sorted_prices[-1]` for bids, `sorted_prices[0]` for asks. New levels inserted via `bisect.insort` in O(log P). Exhausted levels removed via `bisect.bisect_left` + `del` in O(log P + P).

### Global order index — `dict[int, Order]`
O(1) lookup for cancel, modify, and status queries.

### Order history — `deque[Order]`
Chronological log. `view_last_orders(n)` uses `itertools.islice(reversed(...), n)` — O(n), not O(N).

### Trade tape — `list[Trade]`
Append-only. Reversed iteration for newest-first retrieval.

---

## 2. Rejected Alternatives

| Alternative | Why rejected |
|---|---|
| `heapq` for prices | No efficient arbitrary deletion — cancelling or exhausting a mid-book price level requires O(P) rebuild or lazy deletion that pollutes queries. |
| `sortedcontainers.SortedList` | Third-party; challenge requires standard library only. |
| Self-balancing BST | Not in standard library; ~300 lines to implement for marginal gain when P is small. |
| `list` for order queues | `list.pop(0)` is O(n) vs `deque.popleft()` O(1). |

**Trade-off accepted:** `bisect` list shift is O(P) worst-case, but P (distinct price levels) is far smaller than total order count, so this is negligible in practice.

---

## 3. Complexity

| Operation | Complexity | Notes |
|---|---|---|
| `submit_order` | O(log P + K) | bisect insert + K fills |
| `cancel_order` | O(1) amortised | Lazy mark + O(1) volume decrement; level cleanup O(log P + P) when volume hits 0 |
| `modify_order` | O(Q + log P) | Eager deque removal (Q = orders at that price) + bisect re-insert; qty-decrease-only is O(1) |
| `get_best_bid/ask` | O(1) | Index into sorted list + volume dict lookup |
| `get_spread` | O(1) | Two best-price lookups |
| `get_market_depth` | O(levels) | Walk sorted prices, read volume dict |
| `get_volume_at_price` | O(1) | Volume dict lookup |
| `view_last_orders` | O(n) | `itertools.islice` on reversed deque |
| `view_largest/smallest` | O(N log n) | `heapq.nlargest/nsmallest` |
| `get_trade_history` | O(T) | Reversed iteration, bounded by limit |
| `get_order_status` | O(1) + O(F) | Dict lookup + fill list copy |

*P = price levels, K = fills, Q = orders at one price, N = total orders, n = requested count, T = trades, F = fills on one order.*

---

## 4. Key Design Decisions

**Hybrid cancellation.** Orders are lazily marked cancelled (O(1) volume decrement), not eagerly removed from their deque (which would be O(Q)). The matching loop skips and pops cancelled entries. Price levels are eagerly cleaned when volume hits zero, keeping `best_price()` correct.

**FOK feasibility check.** Read-only walk of the contra side before matching — no state mutation if the order can't fill. Cost is O(K), same as matching itself.

**`_level_volume` for O(1) volume.** A separate `dict[float, int]` avoids iterating the deque to compute volume. Updated on every add (+qty), cancel (-qty), fill (-qty), and modify (±delta).

**`_remove_price_level` uses `bisect.bisect_left` + `del`.** O(log P) binary search to find the index, then O(P) shift. More efficient than `list.remove` which does O(P) linear scan first.

---

## 5. Edge Cases

| Scenario | Behaviour |
|---|---|
| Market order on empty book | Cancelled immediately |
| FOK with insufficient liquidity | Rejected before matching; no fills, no mutation |
| IOC partial fill | Filled portion executes; remainder cancelled |
| Cancel filled/cancelled order | Returns False |
| Modify below filled quantity | Returns False |
| Qty decrease modify | Priority preserved; volume dict updated in-place |
| Price/qty-increase modify | Loses priority; re-queued with new timestamp |
| Self-trade | Allowed per spec |
| Unknown asset/order ID | Returns None or 0; no errors |
| Stale cancelled entries in deque | Skipped and popped during matching |