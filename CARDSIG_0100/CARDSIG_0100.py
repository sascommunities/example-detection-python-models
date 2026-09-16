# Copyright © 2025, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

from datetime import datetime

class CARDSIG_0100:

    # Required method that contains default field values when the class is initialized
    def __init__(self):
        self.amounts = [None] * 10
        self.datetimes = [None] * 10
        self.curr_sig_length = 0

    # Mandatory function to resolve Signature keys
    # Should return a lookup key or None
    def resolve(self, hqo_card_num, rqo_tran_dttm, tca_mod_amt):
        if hqo_card_num:
            return hqo_card_num
        return None

    # Mandatory method to update Signature based on current transaction
    def update(self, hqo_card_num, rqo_tran_dttm, tca_mod_amt) -> bool:
        dt = datetime.fromisoformat(rqo_tran_dttm)
        if self.curr_sig_length == 10:
            self.amounts.pop(0)
            self.amounts.append(tca_mod_amt)
            self.datetimes.pop(0)
            self.datetimes.append(dt)
        else:
            self.amounts[self.curr_sig_length] = tca_mod_amt
            self.datetimes[self.curr_sig_length] = dt
            self.curr_sig_length += 1
        # Must return a Boolean to indicate whether updates should be persisted in the DB or not
        return True

    # Mandatory method to serialize data members into a binary blob
    def serialize(self) -> bytes:
        bBlob = pickle.dumps(self)
        return bBlob

    # Mandatory method to deserialize binary blob into data members
    def deserialize(self, bBlob: bytes):
        incoming_sig = pickle.loads(bBlob)
        self.__dict__.update(incoming_sig.__dict__)

    def extra_method(self):
        return "wow what a method {0}".format(self.datetimes)
