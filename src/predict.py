import os
import logging
from typing import Any, List, Optional, Union

import torch
import librosa
import pickle
import gradio as gr
import soundfile as sf

from espnet_emotion.model import EmotionEspnetModel, ESPNetEmotionClassifier
from espnet_emotion.plutchik_vad import get_emotion_from_vad

from visual.pyplutchik import plutchik
from visual.mappings import PLUTCHIK_PLOT_MAPPING, PLUTCHIK_PLOT_VALUES

from config import config, BaseConfig

''' CPU/GPU Configurations '''
if torch.cuda.is_available():
    DEVICE = [0]  # use 0th CUDA device
    ACCELERATOR = 'gpu'
else:
    DEVICE = 1
    ACCELERATOR = 'cpu'

MAP_LOCATION: str = torch.device('cuda:{}'.format(DEVICE[0]) if ACCELERATOR == 'gpu' else 'cpu')

''' Gradio Input/Output Configurations '''
inputs: Union[str, gr.Audio] = gr.Audio(source='upload', type='filepath')
# inputs: Union[str, gr.inputs.Audio] = gr.inputs.Audio(source='upload', type='filepath')
outputs: List[Union[str, gr.Plot]] = ['text', gr.Plot()]

''' Helper functions '''
def initialize_emo_model(cfg: BaseConfig) -> EmotionEspnetModel:

    classifier = ESPNetEmotionClassifier(cfg.espnet_cfg_filepath, cfg.er_model_filepath)
    classifier.eval()
    emo_model = EmotionEspnetModel(classifier)

    return emo_model

def initialize_vad_model(cfg: BaseConfig):

    model_vad2xy = pickle.load(open(cfg.vad_model_filepath, 'rb'))

    return model_vad2xy

def draw_plutchik(emotion_label: str):

    return plutchik(
        scores=PLUTCHIK_PLOT_VALUES, 
        highlight_emotions=PLUTCHIK_PLOT_MAPPING[emotion_label][0],
        highlight_intensity=PLUTCHIK_PLOT_MAPPING[emotion_label][1],
        sub_emotion=emotion_label
        )

''' Initialize models '''
emo_model = initialize_emo_model(config)
vad_model = initialize_vad_model(config)

''' Main prediction function '''
def predict(audio_path: str) -> str:

    with torch.no_grad():

        emotion_from_espnet, v, a, d = emo_model.forward(audio_path)
        emotion_label = get_emotion_from_vad(v, a, d, vad_model)

    return emotion_label, draw_plutchik(emotion_label)
