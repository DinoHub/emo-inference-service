import pandas as pd
import pickle 
import numpy as np
from typing import Any


def normalize_learned_vad(v: float, a: float, d: float):

    regression_min = [-0.88493859 , 0.3582832 , -0.47001082] 
    regression_max = [0.07790179,  0.55874855 , 0.61162631]

    v_new = (2* ( (v - regression_min[0] / (regression_max[0] - regression_min[0])) -0.5)) # + np.random.normal(-0.5,0.5,1)[0] #-1 , 1 noise of 0, 0.2
    a_new = (a - regression_min[1] / (regression_max[1] - regression_min[1])) # + np.random.normal(0,0.2,1)[0] #0, 1 + # noise of 0,0.1
    d_new = (2* ( (d - regression_min[2] / (regression_max[2] - regression_min[2])) -0.5)) #+ np.random.normal(-0.5,0.5,1)[0] #-1,  1

    return v_new, a_new, d_new

def updated_plutchik():

    def add_row(radius, angle, v, a, d, emo):
        df.loc[len(df)] = [radius, angle, v, a, d, emo]

    df = pd.DataFrame(columns=['radius', 'angle', 'v', 'a', 'd', 'emotion'])
        
    # first radius 
    add_row(1, 0, -0.31, 0.48, 0.03, 'ecstasy')  # ecstasy 
    add_row(1, 45, -0.43, 0.49, 0.17, 'vigilance') # vigilance 
    add_row(1, 90, -0.54, 0.49, 0.25, 'rage') # rage
    add_row(1, 135, -0.56, 0.47, 0.23, 'loathing') # loathing
    add_row(1, 180, -0.5, 0.44, 0.11, 'grief') # grief
    add_row(1, 225, -0.37, 0.43, -0.03, 'amazament') # amazement
    add_row(1, 270, -0.27, 0.43, -0.11, 'terror') # terror
    add_row(1, 315, -0.24, 0.45, -0.09, 'admiration') # admiration

    # second radius
    add_row(2, 0, -0.22, 0.5, -0.02, 'joy') # joy 
    add_row(2, 45, -0.46, 0.53, 0.26, 'anticipation') # anticipation
    add_row(2, 90, -0.67, 0.52, 0.43, 'anger') # anger
    add_row(2, 135, -0.72, 0.47, 0.39, 'disgust') # disgust
    add_row(2, 180, -0.59, 0.42, 0.16, 'sadness') # sadness
    add_row(2, 225, -0.34, 0.39, -0.12, 'surprise') # surprise
    add_row(2, 270, -0.14, 0.4, -0.29, 'fear') # fear
    add_row(2, 315, -0.08, 0.44, -0.25, 'trust') # trust (although copilot gave D a -0.125)

    # third radius
    add_row(3, 0, -0.13, 0.51, -0.06, 'serenity') # serenity
    add_row(3, 45, -0.49, 0.56, 0.36, 'interest') # interest
    add_row(3, 90, -0.81, 0.54, 0.61, 'annoyance') # annoyance
    add_row(3, 135, -0.88, 0.48, 0.55, 'boredom') # boredom
    add_row(3, 180, -0.68, 0.4, 0.2, 'pensiveness') # pensive
    add_row(3, 225, -0.32, 0.36, -0.22, 'distraction') # distraction
    add_row(3, 270, -0, 0.37, -0.47, 'apprehension') # apprehension
    add_row(3, 315, 0.08, 0.44, -0.4, 'acceptance') # acceptance (note that dominance from copilot was 0)

    return df

def get_emotion_from_vad(v: float, a: float, d: float, model_vad2xy: Any, plutchik_df: pd.DataFrame = updated_plutchik()):

    v,a,d = normalize_learned_vad(v,a,d)
    test_sample = np.array([v,a,d]).reshape(1,-1)
    xy = model_vad2xy.predict(test_sample)

    dists = []
    emotions = []
    for _,  row in plutchik_df.iterrows():
        r1, a1 = row['radius'], row['angle']
        x_1 = r1 * np.cos(np.deg2rad(a1))
        y_1 = r1 * np.sin(np.deg2rad(a1))
        x_2, y_2 = xy[0][0], xy[0][1]
        dist = np.sqrt((x_2-x_1)**2 + (y_2-y_1)**2)
        dists.append(dist)
        emotions.append(row['emotion'])
    dists, emotions = np.array(dists), np.array(emotions)

    min_dist = np.min(dists)
    # if min_dist > 10: emotion='neutral1'
    emotion=emotions[np.argmin(dist)]

    return emotion 
