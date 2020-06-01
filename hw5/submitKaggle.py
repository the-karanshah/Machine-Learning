import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pred = pd.read_csv('predTestLabels.csv', header=None)


def submission(y_pred):
    save_file = pd.DataFrame(columns=["Id","Prediction"])
    id = np.array([idx for idx in range(1, len(pred))])
    save_file["ImgId"] = id
    save_file["Prediction"]= y_pred
    save_file.to_csv("submission2.csv", index=0)

submission(pred)