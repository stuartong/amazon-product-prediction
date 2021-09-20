import pandas as pd
import numpy as np
import os
import yaml

def detect_fake_reviews():
    with open("params.yaml", "r") as file:
        params= yaml.safe_load(file)

    