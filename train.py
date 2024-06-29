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


begin_fc=120
end_fc=8658
depths=[2.5,3,3.5]
electroids_num=12
exp=1

def predict(sig,fs):
    #给定信号，频率输出通过模型预测的句子
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

def depth_adjust(cfs,depth=0):
    #move cfs to simulate different inplanting depth in insertion
    _depths=np.array([depth]*electroids_num)
    adjusted_depth=(35 / 2.1) * np.log10((np.array(cfs)/ 165.4) + 0.88)+_depths
    return 165.4 * (10**(2.1 * adjusted_depth / 35) - 0.88)

def apply_normal_variation(cfs):
#这里要改，改成cfs，因为cfs和bands有鲜明的对应关系，而且cfs容易变动，注意上限下限
    #bands=generate_bands(min(cfs),max(cfs),len(cfs))
    m=np.array([168]*(electroids_num))#经验矫正项，用于把cfs拉平成一个真正的直线
    cuts=list(np.log(cfs+m))
    # 计算变动的标准差，即原数值的百分比
    std_devs = [0.03]*electroids_num
    # 生成正态分布的随机变动
    variations = np.random.normal(0, std_devs)
    random_choose = random.sample(range(0, electroids_num), int(electroids_num/3))
    variations=[variations[i] if i in random_choose else 0 for i in range(electroids_num)]
    # 应用变动
    cuts = np.array(cuts) + np.array(variations)
    new_cfs=np.exp(cuts)-m
    if max(new_cfs)>=end_fc:
        new_cfs[-1]=end_fc-100
        new_cfs.sort()
    return new_cfs

def cfs_to_bands(cfs):
    m=np.array([168]*(electroids_num+1))#经验矫正项，用于把cfs拉平成一个真正的直线
    cuts=list(np.log(cfs+m[1:]))
    cuts=np.array([2*cuts[0]-cuts[1]]+cuts+[2*cuts[-1]-cuts[-2]])
    block=np.exp([(cuts[i]+cuts[i+1])/2 for i in range(0,electroids_num+1)])-m
    bands=[[block[i],block[i+1]] for i in range(0,len(block)-1)]
    return np.array(bands)

#read sound files
def read_sound():
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
    # load pretrained model
    #read sound data
    (sentenses,sig_ls, fs)=read_sound()

    cfs=generate_cfs(begin_fc, end_fc, electroids_num)
    record=[[cfs,3374.5793]]
    f=open("record6.txt","w")

    start = time.perf_counter() # 记录开始时间
    if exp==0:
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
            
            


    if exp==1:
        cfs=generate_cfs(begin_fc, end_fc, electroids_num)
        record=[]
        for depth in depths:
            new_cfs=depth_adjust(cfs,depth)
            loss=0
            bands=cfs_to_bands(new_cfs)
            for i in range(0,70):
                try:
                    sig_out = spyral.spyral(sig_ls[i], fs, new_cfs, 80, -20, analysis_cutoffs=np.array(bands))
                    loss += get_loss(sentenses[i],sig_out,fs)
                except:
                    sig_out = 0
                    loss=1000000
            record.append([depth,loss])
            f.write(str(depth))
            f.write("loss: "+str(loss)+"\n")
            print(str(depth))
            print("loss: "+str(loss))

    end = time.perf_counter()   # 记录结束时间
    elapsed = end - start        # 计算经过的时间（单位为秒)
    print("程序运行时间：", elapsed/60)
    print("exp"+str(exp))
    f.close()


