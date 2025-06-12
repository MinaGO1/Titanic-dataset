import sys
sys.dont_write_bytecode = True

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
from sklearn.preprocessing import RobustScaler , LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')