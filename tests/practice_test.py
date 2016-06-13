#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as platform
pd.set_option('max_columns', 50)

url = "http://stat.columbia.edu/~gelman/arm/examples/arsenic/wells.dat"
wells = pd.read_table(url, sep='\s+', header=0, index_col=0)
print wells.head()