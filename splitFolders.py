# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import splitfolders

originPath = 'flowers'
finalPath  = 'flowers\splited'


splitfolders.ratio(originPath,finalPath, seed=40,ratio=(0.6,0.2,0.2),group_prefix=None)