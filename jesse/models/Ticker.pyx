# import peewee
from jesse.services.db import database
import random
def uuid4():
  return '%032x' % random.getrandbits(128)

class Ticker():
    id = uuid4() #peewee.UUIDField(primary_key=True)
    # timestamp in milliseconds
    timestamp = int #peewee.BigIntegerField()
    # the latest trades price
    last_price = float #peewee.FloatField()
    # the trading volume in the last 24 hours
    volume = float #peewee.FloatField()
    # the highest price in the last 24 hours
    high_price = float #peewee.FloatField()
    # the lowest price in the last 24 hours
    low_price = float #peewee.FloatField()
    symbol = str #peewee.CharField()
    exchange = str #peewee.CharField()

    class Meta:

        database = database.db
        indexes = ((('exchange', 'symbol', 'timestamp'), True),)

    def __init__(self, attributes: dict = None, **kwargs) -> None:
        # peewee.Model.__init__(self, attributes=attributes, **kwargs)

        if attributes is None:
            attributes = {}

        for a, value in attributes.items():
            setattr(self, a, value)

