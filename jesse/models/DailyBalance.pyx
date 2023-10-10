# import peewee
from jesse.services.db import database
import random
# def uuid4():
  # return '%032x' % random.getrandbits(128)
import ruuid as uuid
if database.is_closed():
    database.open_connection()


class DailyBalance():
    id = uuid.uuid4()
    timestamp = int
    identifier = str = None
    exchange = str
    asset = str
    balance = float

    class Meta:
        # from jesse.services.db import database

        database = database.db
        indexes = (
            (('identifier', 'exchange', 'asset', 'timestamp'), True),
            (('identifier', 'exchange'), False),
        )

    def __init__(self, attributes: dict = None, **kwargs) -> None:
        # peewee.Model.__init__(self, attributes=attributes, **kwargs)

        if attributes is None:
            attributes = {}

        for a, value in attributes.items():
            setattr(self, a, value)


# if database is open, create the table
# if database.is_open():
    # DailyBalance.create_table()
