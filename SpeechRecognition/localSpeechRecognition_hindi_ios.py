#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 23:04:02 2022

@author: vishnu.vithala
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 22:27:48 2022

@author: vishnu.vithala
"""

import torch
from torch import Tensor
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchaudio
from torchaudio.models.wav2vec2.utils.import_huggingface import import_huggingface_model
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
import io
from pydub import AudioSegment



tokenizer = Wav2Vec2Processor.from_pretrained('theainerd/Wav2Vec2-large-xlsr-hindi')
model = Wav2Vec2ForCTC.from_pretrained('theainerd/Wav2Vec2-large-xlsr-hindi')

model = import_huggingface_model(model)
model.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
model = model.eval()
r = sr.Recognizer()

quantized_model = torch.quantization.quantize_dynamic(
     model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)


scripted_model = torch.jit.script(quantized_model)
optimized_model = optimize_for_mobile(scripted_model)

# with sr.Microphone(sample_rate=16000) as source :
#     print('you can start speaking now')
#     while True:
#         audio = r.listen(source)
#         data = io.BytesIO(audio.get_wav_data()) #arrayofbytes
#         clip = AudioSegment.from_file(data) #numpy array
#         x= torch.FloatTensor(clip.get_array_of_samples()) 
        
#         inputs = tokenizer(x, sampling_rate=16000, return_tensors='pt',padding='longest').input_values
        
#         #logits = model(inputs).logits   , https://github.com/huggingface/transformers/issues/7638
#         logits = model(input, return_dict=True).logits

#         tokens = torch.argmax(logits,axis = -1)
#         text = tokenizer.batch_decode(tokens) #token into a string , 
        
#         print('You said:', str(text).lower())
        



optimized_model._save_for_lite_interpreter("SpeechRecognition/wav2vec2_hindi.ptl")
