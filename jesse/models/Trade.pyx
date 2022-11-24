# import peewee
import random
def uuid4():
  return '%032x' % random.getrandbits(128)
from jesse.services.db import database

class Trade():
    id = uuid4()
    # timestamp in milliseconds
    timestamp = int

    price = float

    buy_qty = float
    sell_qty = float

    buy_count = int
    sell_count = int

    symbol = str
    exchange = str

    class Meta:

        database = database.db
        indexes = ((('exchange', 'symbol', 'timestamp'), True),)

    def __init__(self, attributes: dict = None, **kwargs) -> None:
        # peewee.Model.__init__(self, attributes=attributes, **kwargs)

        if attributes is None:
            attributes = {}

        for a, value in attributes.items():
            setattr(self, a, value)
