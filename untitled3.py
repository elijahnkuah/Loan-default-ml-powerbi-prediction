from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import catboost as cat_
import seaborn as sns
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer as Imputer
from sklearn import preprocessing
import re
import timeit
import random
import joblib

from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return ("This is Elijah !")

@app.route("/try")
def try_work():
    return ("We have to gain it by the Grace of God")

app.run()
