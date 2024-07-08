import spyral
import pysndfile
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pandas as pd
from tools import generate_bands,generate_cfs
import numpy as np
import random
import time

#This program is for MCMC algorithm
#Please put the tools on the same dictionay with this file

#start and end frequency for the CI
begin_fc=120
end_fc=8658
electroids_num=12

#subfunction, ignore them
def predict(sig,fs):
    #predict the sentences from input signal.
    input_values = processor(sig, sampling_rate=fs, return_tensors="pt").input_values
    # INFERENCE
    # retrieve logits & take argmax
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # transcribe
    transcription = processor.decode(predicted_ids[0])
    # FINE-TUNE
    return (transcription)

def get_loss(sentense, sig, fs):
    input_values = processor(sig, sampling_rate=fs, return_tensors="pt").input_values
    # encode labels
    with processor.as_target_processor():
        labels = processor(sentense, return_tensors="pt").input_ids

    # compute loss by passing labels
    loss = model(input_values, labels=labels).loss
    return loss

def apply_normal_variation(cfs):
    #bands=generate_bands(min(cfs),max(cfs),len(cfs))
    #168 is a empirical correction term, used to flatten the CFS into a true straight line.
    m=np.array([168]*(electroids_num))
    cuts=list(np.log(cfs+m))
    # Calculate the standard deviation of the variation, expressed as a percentage of the original values.
    std_devs = [0.03]*electroids_num
    # Generate random variations following a normal distribution.
    variations = np.random.normal(0, std_devs)
    random_choose = random.sample(range(0, electroids_num), int(electroids_num/3))
    variations=[variations[i] if i in random_choose else 0 for i in range(electroids_num)]
    # apply the variations
    cuts = np.array(cuts) + np.array(variations)
    new_cfs=np.exp(cuts)-m
    if max(new_cfs)>=end_fc:
        new_cfs[-1]=end_fc-100
        new_cfs.sort()
    return new_cfs

def cfs_to_bands(cfs):
    #168 is a empirical correction term, used to flatten the CFS into a true straight line.
    m=np.array([168]*(electroids_num+1))
    cuts=list(np.log(cfs+m[1:]))
    cuts=np.array([2*cuts[0]-cuts[1]]+cuts+[2*cuts[-1]-cuts[-2]])
    block=np.exp([(cuts[i]+cuts[i+1])/2 for i in range(0,electroids_num+1)])-m
    bands=[[block[i],block[i+1]] for i in range(0,len(block)-1)]
    return np.array(bands)

#read sound files
def read_sound():
    #read the sound file, in mp3 form
    librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    sig_ls=[]
    sentenses=[]
    for i in range(0,70):
        (sig,fs,enc) = pysndfile.sndio.read(librispeech_samples_ds[i]["file"])
        sig_ls.append(sig)
        sentenses.append(predict(sig,fs))
    return sentenses, sig_ls, fs

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

if __name__=="__main__":
    #read sound data
    (sentenses,sig_ls, fs)=read_sound()
    #Same as the original function in vocoder
    cfs=generate_cfs(begin_fc, end_fc, electroids_num)

    #Initialization record, 3374.5793 is the default loss under CFS.
    record=[[cfs,3374.5793]]
    f=open("record6.txt","w")

    start = time.perf_counter() # record the time
    #exp=0 is the MCMC algorithm
    #mcmc
    candidate_cfs=cfs
    iter=0
    while (iter<=100):
        loss=0
        bands=cfs_to_bands(candidate_cfs)
        for i in range(0,70):
            try:
                sig_out = spyral.spyral(sig_ls[i], fs, candidate_cfs, 80, -20, analysis_cutoffs=np.array(bands))
                loss += get_loss(sentenses[i],sig_out,fs)
            except:
                sig_out = 0
                loss=1000000
                break
        if loss!=1000000:
            if random.uniform(0.98, 1)<record[-1][1]/loss:
               cfs=candidate_cfs
            record.append([candidate_cfs,loss])
            f.write(str(list(candidate_cfs)))
            f.write("loss: "+str(loss)+"\n")
            
            print(str(list(candidate_cfs)))
            print("loss: "+str(loss))
            print("iter: "+str(iter-1))
            candidate_cfs=apply_normal_variation(cfs)
        else:
            candidate_cfs=record[-2][0]
        iter+=1
            
    end = time.perf_counter()  
    elapsed = end - start 
    print("running time:", elapsed/60)
    f.close()


