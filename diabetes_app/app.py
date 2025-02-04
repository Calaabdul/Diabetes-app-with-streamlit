import pickle
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd


app = Flask(__name__)

with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)