import arrow
import random
# def uuid4():
  # s = '%032x' % random.getrandbits(128)
  # return s[0:8]+'-'+s[8:12]+'-4'+s[13:16]+'-'+s[16:20]+'-'+s[20:32]
import ruuid as uuid

class AppState:
    def __init__(self):
        self.time = arrow.utcnow().int_timestamp * 1000
        self.starting_time = None
        self.daily_balance = []

        # used as placeholders for detecting open trades metrics
        self.total_open_trades = 0
        self.total_open_pl = 0
        self.total_liquidations = 0

        self.session_id = ''
        self.session_info = {}
		
    def set_session_id(self) -> None:
        """
        Generated and sets session_id. Used to prevent overriding of the session_id
        """
        if self.session_id == '':
            self.session_id = uuid.uuid4()
            
    def clear(self) -> None:
        #resets everything
        self.daily_balance = []
