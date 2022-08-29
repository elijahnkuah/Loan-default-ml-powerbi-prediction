# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 08:44:18 2021

@author: Elijah_Nkuah
"""

OldRange = (OldMax - OldMin)
if (OldRange == 0)
    NewValue = NewMin
else
{
    NewRange = (NewMax - NewMin)new_value  
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
}


old_value = 10000
old_min = -16000
old_max = 16000
new_min = 0
new_max = 100

new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min


old_value = 12.468708716523489
old_min = 0
old_max = 100
new_min = 300
new_max = 850

new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
new_value