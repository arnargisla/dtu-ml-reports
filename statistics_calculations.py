#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 02:30:02 2017

@author: Onat1
"""

import numpy as np
from matplotlib.pyplot import boxplot, title, show
import xlrd

doc = xlrd.open_workbook('murder.xls').sheet_by_index(0)
attributeNames = doc.row_values(0, 0, doc.ncols)
x = np.mat(np.empty((doc.nrows-1, doc.ncols)))
for i, col_id in enumerate(range(0, doc.ncols)):
    x[:, i] = np.mat(doc.col_values(col_id, 1, doc.nrows)).T


for i in range(0,doc.ncols):
    xi_array = np.asarray(x[:, i].T)[0]
    print(attributeNames[i])
    print("Mean: ", xi_array.mean())
    print('Standard Deviation:', xi_array.std(ddof=1))
    print('Median:', np.median(xi_array))
    print('Range:', xi_array.max()-xi_array.min())
    boxplot(xi_array)
    title(attributeNames[i] + " - boxplot")
    show()