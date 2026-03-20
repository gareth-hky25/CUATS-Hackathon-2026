from typing import Optional


class Order:

    def __init__(self, order_id, order_owner, order_type, side, asset, quantity,
                 price: Optional[float] = None, timestamp=0):
        self.order_id = order_id
        self.order_owner = order_owner
        self.order_type = order_type
        self.side = side
        self.asset = asset
        self.quantity = quantity
        self.price = price
        self.filled_quantity = 0
        self.remaining_quantity = quantity
        self.status = "pending"
        self.timestamp = timestamp
        self.trades = []

    def __str__(self):
        return (
            f"Order(order_id={self.order_id}, order_owner={self.order_owner!r}, "
            f"order_type={self.order_type!r}, side={self.side!r}, "
            f"asset={self.asset!r}, quantity={self.quantity}, "
            f"price={self.price}, filled={self.filled_quantity}, "
            f"remaining={self.remaining_quantity}, status={self.status!r})"
        )

    def __repr__(self):
        return self.__str__()


class Trade:

    __slots__ = ("asset", "price", "quantity", "buy_order_id", "sell_order_id")

    def __init__(self, asset, price, quantity, buy_order_id, sell_order_id):
        self.asset = asset
        self.price = price
        self.quantity = quantity
        self.buy_order_id = buy_order_id
        self.sell_order_id = sell_order_id

    def __str__(self):
        return (
            f"Trade(asset={self.asset!r}, price={self.price}, "
            f"quantity={self.quantity}, buy_order_id={self.buy_order_id}, "
            f"sell_order_id={self.sell_order_id})"
        )

    def __repr__(self):
        return self.__str__()


class OrderResult:

    __slots__ = ("order_id", "status", "fills", "remaining_quantity")

    def __init__(self, order_id, status, fills=None, remaining_quantity=0):
        self.order_id = order_id
        self.status = status
        self.fills = fills if fills is not None else []
        self.remaining_quantity = remaining_quantity

    def __str__(self):
        return (
            f"OrderResult(order_id={self.order_id}, status={self.status!r}, "
            f"fills={len(self.fills)}, remaining={self.remaining_quantity})"
        )

    def __repr__(self):
        return self.__str__()


class OrderStatus:

    __slots__ = ("order_id", "status", "filled_quantity", "remaining_quantity", "trades")

    def __init__(self, order_id, status, filled_quantity, remaining_quantity, trades=None):
        self.order_id = order_id
        self.status = status
        self.filled_quantity = filled_quantity
        self.remaining_quantity = remaining_quantity
        self.trades = trades if trades is not None else []

    def __str__(self):
        return (
            f"OrderStatus(order_id={self.order_id}, status={self.status!r}, "
            f"filled={self.filled_quantity}, remaining={self.remaining_quantity}, "
            f"trades={len(self.trades)})"
        )

    def __repr__(self):
        return self.__str__()