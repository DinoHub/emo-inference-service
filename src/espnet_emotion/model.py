import sys 
import torch
import torchaudio 
import numpy as np
import soundfile as sf 
from espnet2.bin.er_inference import Speech2Text

class ESPNetEmotionClassifier(torch.nn.Module):
    def __init__(self, espnet_cfg: str, er_model_filepath: str, token_type: str = 'word', nbest: int = 1):

        super(ESPNetEmotionClassifier, self).__init__()

        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        bundle = torchaudio.pipelines.HUBERT_LARGE
        self.feature_extractor = bundle.get_model().to(self.device)
        self.inference = Speech2Text(espnet_cfg, er_model_filepath ,token_type, bpemodel=None, device=self.device)
        self.nbest = nbest
        self.limit = 160000 #( Sampling Rate = 16000 * len in seconds=10s )

    def forward(self, audio_filepath: str):

        audio, sr = sf.read(audio_filepath)
        # normalize to mono channel
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)
        
        audio = torch.from_numpy(audio.reshape(1,-1)).to(self.device).float()
        audio_len = audio.shape[-1]
        with torch.no_grad():
            if audio_len > self.limit:
                feats = []
                for i in range(0,audio_len,self.limit):
                    audio_sam = audio[:,i:i+self.limit]
                    audio_sam1= audio[:,i:]
                    x,_ = self.feature_extractor(audio_sam) if i + self.limit < audio_len else self.feature_extractor(audio_sam1)
                    x = x[-1]
                    feats.append(x)
                feats = torch.cat(feats)
            else:
                feats , _ = self.feature_extractor(audio)
                feats = feats[-1] 

            batch = {"speech": feats }
            results = self.inference(**batch)
            
            emotion_cts = None 
            emotion_disc = None 
            disc_prob = None 

            text, token, token_int, score, emo_out = results[0]
            if emo_out is not None:
                    emotion_cts = emo_out
            if text is not None:
                emotion_disc  = ("").join(text).replace(" ", "")
                disc_prob = score[0]
            return emotion_disc, disc_prob, emotion_cts

class EmotionEspnetModel:
    def __init__(self, classifier: ESPNetEmotionClassifier):

        super(EmotionEspnetModel, self).__init__()

        self.classifier = classifier
        self.emotion_mapping = {'<neu>': 'Neutral', '<sad>': 'Sad', '<ang>': 'Anger', '<hap>': 'Happy', '<dis>': 'Disappointment'}

    def normalize_vad(self, vad):

        v= 2*((vad[0]/7) - 0.5 ) # -1, 1
        d= 2*((vad[0]/7) - 0.5 ) # -1, 1
        a= vad[0]/7 # 0, 1

        return v,a,d

    def forward(self, audio_filepath: str):

        emotion_disc, disc_prob, emotion_cts = self.classifier(audio_filepath)
        emo = self.emotion_mapping[emotion_disc]
        v, a, d = emotion_cts
        v, a, d = self.normalize_vad([v,a,d])

        return emo, v, a, d

if __name__ == "__main__":

    args = sys.argv
    audio_filepath = args[1]
    
    esp = ESPNetEmotionClassifier(espnet_cfg='misc/espnet_configs/config.yaml', er_model_filepath='models/valid.ccc.ave_10best.pth')
    model = EmotionEspnetModel(esp)

    emotion, v, a, d = model.forward(audio_filepath)
    print(emotion, v, a, d)
