# Copyright © 2025, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import math
import numpy as np
import os
import pickle

from mlserver import MLModel
from typing import List
from pythonmodelruntime.signatures import init_signatures, get_signatures_path, register_metrics, SignatureInterface
from pythonmodelruntime.utils import decode_and_extract_args


# User provides load and predict methods
class PYTEST_0200(MLModel):

    async def load(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tree_model.pkl'), 'rb') as _pFile:
            self.model = pickle.load(_pFile)

        risktable = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mcc_risk.csv')
        reader = csv.reader(open(risktable, 'r'))
        self.mcc_risk = {}
        for row in reader:
            m, v = row
            self.mcc_risk[str(m).zfill(4)] = float(v)

        # Models using signatures must initialize the signature classes using provided utility functions
        self.signatures = init_signatures(get_signatures_path())

        # Optional: register signature processing metrics with MLServer metrics endpoint
        register_metrics()

    # MLServer provides @decode_args normally, which decodes the inference input into an array.
    # @decode_and_extract_args is a decorator provided by the pythonmodelruntime package
    # that extracts the first element of the input array.
    @decode_and_extract_args
    async def predict(
        self,
        tca_mod_amt: np.ndarray,
        hct_mer_mcc: List[str],
        hct_term_cntry_code: List[str],
        ucm_pos: List[str],
        CARDSIG_0100: List[SignatureInterface]  # SignatureInterface allows us to pass signatures as Python objects
    ) -> np.ndarray:
        log_amount = math.log10(tca_mod_amt + 0.0001)
        is_foreign = 1

        if hct_term_cntry_code  == '840':
            is_foreign = 0

        pos_00 = 0
        pos_01 = 0
        pos_02 = 0
        pos_05 = 0
        pos_07 = 0
        pos_80 = 0
        pos_81 = 0
        pos_8F = 0
        pos_90 = 0
        pos_91 = 0
        pos_AP = 0
        pos_DP = 0
        pos_GM = 0
        pos_HC = 0
        pos_MM = 0
        pos_SP = 0

        if ucm_pos == '00':
            pos_00 = 1
        elif ucm_pos == '01':
            pos_01 = 1
        elif ucm_pos == '02':
            pos_02 = 1
        elif ucm_pos == '05':
            pos_05 = 1
        elif ucm_pos == '07':
            pos_07 = 1
        elif ucm_pos == '80':
            pos_80 = 1
        elif ucm_pos == '81':
            pos_81 = 1
        elif ucm_pos == '8F':
            pos_8F = 1
        elif ucm_pos == '90':
            pos_90 = 1
        elif ucm_pos == '91':
            pos_91 = 1
        elif ucm_pos == 'AP':
            pos_AP = 1
        elif ucm_pos == 'DP':
            pos_DP = 1
        elif ucm_pos == 'GM':
            pos_GM = 1
        elif ucm_pos == 'HC':
            pos_HC = 1
        elif ucm_pos == 'MM':
            pos_MM = 1
        elif ucm_pos == 'SP':
            pos_SP = 1

        if hct_mer_mcc in self.mcc_risk:
            mr = self.mcc_risk[hct_mer_mcc]
        else:
            mr = 0

        invars = np.array([[log_amount,is_foreign,pos_00,pos_01,pos_02,pos_05,pos_07,pos_80,pos_81,pos_8F,pos_90,pos_91,pos_AP,pos_DP,pos_GM,pos_HC,pos_MM,pos_SP,mr]])

        temp_score = self.model.predict_proba(invars)[0][1]
        SCORE = round(temp_score.item()*1000)

        # MLServer expects return type to be an array
        return np.asarray([SCORE])
