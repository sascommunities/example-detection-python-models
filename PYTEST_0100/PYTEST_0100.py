# Copyright © 2025, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import math
import numpy as np
import os
import pickle
import logging

from mlserver import MLModel                                        #### See Remark 1 ###
from typing import List
from pythonmodelruntime.utils import decode_and_extract_args


class PYTEST_0100(MLModel):                                         #### See Remark 2 ###

    async def load(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lr_model_1.sav'), 'rb') as _pFile:
            self.model = pickle.load(_pFile)

        risktable = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mcc_risk.csv')
        reader = csv.reader(open(risktable, 'r'))
        self.MCCRisk_Hash = {}
        for row in reader:
            m, v = row
            self.MCCRisk_Hash[str(m).zfill(4)] = float(v)

    @decode_and_extract_args                                        #### See Remark 3 ###
    async def predict(                                              #### See Remark 4 ###
        self,
        tca_mod_amt: np.ndarray,
        hct_mer_mcc: List[str],
        hct_term_cntry_code: List[str],
        ucm_pos: List[str]) -> np.ndarray:


        logging.debug('Inside PYTEST_0100.predict()')               #### See Remark 5 ###

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
                  pos_dict["GM"], pos_dict["HC"], pos_dict["MM"], pos_dict["SP"]]])

        score = round(self.model.predict_proba(invars)[0][1].item()*1000)

        return np.asarray([score])
