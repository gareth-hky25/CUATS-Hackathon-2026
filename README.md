# CUATS Hacakthon 2026

**1st Place** — CUATS Hackathon 2026, Cambridge University Algorithmic Trading Society

A high-performance limit order book and matching engine built in 24 hours. Scored 100/100 — passing all 92 test cases across correctness, API robustness, multi-asset support, advanced order types, and performance stress tests (10k–50k order streams).

---

## Table of Contents

1. [Architecture](#architecture)
2. [Data Structures](#data-structures)
3. [Complexity](#complexity)
4. [Key Design Decisions](#key-design-decisions)
5. [Order Types](#order-types)
6. [Grading Results](#grading-results)
7. [Project Structure](#project-structure)
8. [Example Usage](#example-usage)

---

## Architecture

```
OrderBook                              ← single public entry point
 └─ _books: dict[str, _AssetBook]      ← one book per asset, created lazily
      └─ _AssetBook
           ├─ bids: _SideBook
           └─ asks: _SideBook
                ├─ price_map:      dict[float, deque[Order]]   ← O(1) price lookup, O(1) FIFO
                ├─ sorted_prices:  list[float]                 ← O(1) best price, O(log P) insert
                └─ _level_volume:  dict[float, int]            ← O(1) volume queries
```

---

## Data Structures

| Structure | Purpose | Why |
|---|---|---|
| `dict[float, deque[Order]]` | Price → FIFO queue | `dict` = O(1) lookup. `deque` = O(1) popleft for time-priority matching. |
| `list[float]` + `bisect` | Sorted active prices | O(1) best price at list endpoints. O(log P) insertion via binary search. |
| `dict[float, int]` | Per-level volume | O(1) volume queries without iterating deques. |
| `dict[int, Order]` | Order ID index | O(1) cancel, modify, status lookup. |

### Rejected Alternatives

| Alternative | Why rejected |
|---|---|
| `heapq` | No efficient arbitrary deletion — problematic for cancels and exhausted price levels. |
| `sortedcontainers.SortedList` | Third-party; challenge requires standard library only. |
| Self-balancing BST (AVL/red-black) | Not in standard library; ~300 lines for marginal gain when P is small. |

**Trade-off:** `bisect.insort` is O(P) for the list shift, but P (distinct price levels) is far smaller than total order count. Grader feedback confirmed this: *"It is a good trade-off of implementation simplicity for asymptotic optimality."*

---

## Complexity

| Operation | Complexity |
|---|---|
| `submit_order` | O(log P + K) |
| `cancel_order` | O(1) amortised |
| `modify_order` | O(Q + log P); qty-decrease is O(1) |
| `get_best_bid/ask` | O(1) |
| `get_spread` | O(1) |
| `get_market_depth` | O(levels) |
| `get_volume_at_price` | O(1) |

---

## Key Design Decisions

**Hybrid cancellation.** Cancelled orders are lazily marked (O(1) volume decrement), not eagerly removed from the deque. The matching loop skips and pops them. Price levels are eagerly cleaned when volume hits zero, keeping `best_price()` correct. Grader feedback: *"The dual cancel strategy is smart: lazy deletion for cancel_order and eager removal only for the modify_order priority-loss path."*

**FOK feasibility check.** Read-only walk of the contra side before matching — no state mutation if the order can't fill. Grader feedback: *"Your FOK feasibility check gives you the fastest FOK runtime we measured across all submissions."*

**Per-level volume tracking.** A `dict[float, int]` updated on every add/cancel/fill/modify eliminates deque iteration for all volume and market data queries.

---

## Order Types

| Type | Behaviour |
|---|---|
| Limit | Match what you can, rest the remainder on the book. |
| Market | Execute immediately at best available prices. Cancel unfilled remainder. |
| Fill-or-Kill | Fill entirely or cancel — no partial fills. |
| Immediate-or-Cancel | Fill what you can now, cancel the rest. |

---

## Grading Results

| Component | Score |
|---|---|
| Base — Correctness (45 tests) | 50 / 50 |
| Base — API Robustness (11 tests) | 10 / 10 |
| Base — Design Docs | 10 / 10 |
| Tier I — Multi-asset & Depth (10 tests) | 10 / 10 |
| Tier II — IOC / FOK / Modify (21 tests) | 10 / 10 |
| Tier III — Performance (5 tests) | 10 / 10 |
| **Total** | **100 / 100** |

---

## Project Structure

```
exchange/
  __init__.py      # exports OrderBook only
  models.py        # Order, Trade, OrderResult, OrderStatus
  orderbook.py     # OrderBook class, matching engine, all API methods
```

**Constraints:** Python standard library only (`bisect`, `collections`, `heapq`, `itertools`, `typing`). No print statements — all data accessible programmatically.

---

## Example Usage

```python
from exchange import OrderBook

ob = OrderBook()

# --- Place orders ---
ob.submit_order(1, "alice", "limit", "sell", "ETHIOPIAN_YIRGACHEFFE", 100, 10.0)
ob.submit_order(2, "bob",   "limit", "buy",  "ETHIOPIAN_YIRGACHEFFE", 50,  10.0)

# --- Check trades ---
trades = ob.get_trade_history("ETHIOPIAN_YIRGACHEFFE")
# 1 trade: 50 units at £10.0

# --- Order status ---
status = ob.get_order_status(1)
# partial fill: 50 filled, 50 remaining

# --- Market data ---
ob.get_best_ask("ETHIOPIAN_YIRGACHEFFE")   # (10.0, 50)
ob.get_best_bid("ETHIOPIAN_YIRGACHEFFE")   # None — bob's order fully filled
ob.get_spread("ETHIOPIAN_YIRGACHEFFE")     # None — no bid side
ob.get_market_depth("ETHIOPIAN_YIRGACHEFFE", levels=3)
# {'bids': [], 'asks': [(10.0, 50)]}

# --- Cancel remaining ---
ob.cancel_order(1)  # True — cancels alice's remaining 50 units

# --- Advanced order types ---
ob.submit_order(3, "charlie", "limit", "sell", "COLOMBIAN_SUPREMO", 100, 5.0)
ob.submit_order(4, "dave",    "fok",   "buy",  "COLOMBIAN_SUPREMO", 200, 5.0)
# FOK rejected — only 100 available, dave wanted 200

ob.submit_order(5, "eve", "ioc", "buy", "COLOMBIAN_SUPREMO", 80, 5.0)
# IOC fills 80, no remainder placed on book

ob.submit_order(6, "frank", "market", "buy", "COLOMBIAN_SUPREMO", 10)
# Market order fills 10 at best available ask (5.0)

# --- Modify an order ---
ob.submit_order(7, "grace", "limit", "buy", "JAMAICAN_BLUE_MOUNTAIN", 100, 20.0)
ob.modify_order(7, new_quantity=80)    # qty decrease — keeps time priority
ob.modify_order(7, new_price=21.0)     # price change — loses time priority

# --- Views ---
ob.view_last_orders(5)       # 5 most recent orders, newest first
ob.view_largest_orders(3)    # 3 largest non-cancelled orders by quantity
ob.view_orderbook()          # full snapshot of all resting orders
```
