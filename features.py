# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:20:48 2021

@author: Elijah_Nkuah
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError


class loan_variables(BaseModel):
    loan_amnt: Optional[float]
    funded_amnt: Optional[float]
    term: Optional[str]
    int_rate: Optional[float]
    emp_length: Optional[str]
    home_ownership: Optional[str]
    annual_inc: Optional[float]
    purpose: Optional[str]
    dti: Optional[float]
    revol_bal: Optional[float]
    revol_util: Optional[float]
    total_acc: Optional[float]
    application_type: Optional[str]
    tot_cur_bal: Optional[float]
    total_rev_hi_lim: Optional[float]
    Monthly_supposed_payment: Optional[float]
    Total_refund: Optional[float]
    Interest_amnt: Optional[float]
    Monthly_income: Optional[float]
   
class Multipleloan_input(BaseModel):
    inputs: List[loan_variables]