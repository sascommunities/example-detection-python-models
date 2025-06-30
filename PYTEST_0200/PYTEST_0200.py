import csv
import math
import numpy as np
import os
import pickle
from datetime import datetime 
import logging

from mlserver import MLModel
from typing import List
from pythonmodelruntime.signatures import SignatureInterface            ### 1 ###
from pythonmodelruntime.utils import decode_and_extract_args


class PYTEST_0200(MLModel):

    async def load(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lr_model_2.sav'), 'rb') as _pFile:
            self.model = pickle.load(_pFile)

        risktable = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mcc_risk.csv')
        reader = csv.reader(open(risktable, 'r'))
        self.MCCRisk_Hash = {}
        for row in reader:
            m, v = row
            self.MCCRisk_Hash[str(m).zfill(4)] = float(v)

    @decode_and_extract_args
    async def predict(
        self,
        tca_mod_amt: np.ndarray,
        hct_mer_mcc: List[str],
        hct_term_cntry_code: List[str],
        ucm_pos: List[str],
        rqo_tran_dttm: List[str], 
        CARDSIG_0100: List[SignatureInterface]) -> np.ndarray:          ### 2 ###
    
        cardsig = CARDSIG_0100                                          ### 3 ###

        # This block is needed to avoid errors during warmup            ### 4 ###
        if cardsig.curr_sig_length == 0:
          return np.asarray([0])

        # Behavioral features                                           ### 5 ###
        # Average spending amount

        logging.debug(cardsig.amounts)

        avg_amount = sum(cardsig.amounts[0:cardsig.curr_sig_length])/cardsig.curr_sig_length
        avg_amount = avg_amount/100;

        # Time since last
     
        if cardsig.curr_sig_length >=2 :
          tsl = (cardsig.datetimes[cardsig.curr_sig_length - 1] - cardsig.datetimes[cardsig.curr_sig_length - 2]).seconds
        else:
          tsl = 0
     
        tsl = tsl / 86400;
     
        # Total spend in last one day
     
        total_spend = 0
    
        dt = datetime.fromisoformat(rqo_tran_dttm)

        for ii, dtt in enumerate(cardsig.datetimes):
          if dtt != None and ((dt - dtt).seconds <= 86400):
            total_spend += cardsig.amounts[ii]
     
        total_spend = total_spend / 1000;

        # Population features

        log_amount = math.log10(tca_mod_amt + 0.0001)
        is_foreign = 1

        if hct_term_cntry_code  == '840':
            is_foreign = 0

        pos_dict = {}
        for pos in ["00","01","02","05","07","80","81","8F","90","91","AP","DP","GM","HC","MM","SP"]:
            pos_dict[pos] = 0
            
        pos_dict[ucm_pos] = 1
            

        if hct_mer_mcc in self.MCCRisk_Hash:
            mcc_risk = self.MCCRisk_Hash[hct_mer_mcc]
        else:
            mcc_risk = 0

        invars = np.array([[log_amount,is_foreign,mcc_risk,
                  pos_dict["00"], pos_dict["01"], pos_dict["02"], pos_dict["05"], 
                  pos_dict["07"], pos_dict["80"], pos_dict["81"], pos_dict["8F"], 
                  pos_dict["90"], pos_dict["91"], pos_dict["AP"], pos_dict["DP"],
                  pos_dict["GM"], pos_dict["HC"], pos_dict["MM"], pos_dict["SP"],
                  avg_amount, tsl, total_spend]])

        score = round(self.model.predict_proba(invars)[0][1].item()*1000)

        return np.asarray([score])
