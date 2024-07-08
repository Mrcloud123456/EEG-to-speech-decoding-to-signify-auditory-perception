import pysndfile
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pandas as pd
import numpy as np
import random
import time
import scipy.signal
from io import BytesIO
from tools import generate_cfs, generate_bands, make_fir_filter

#Here I change a litte in spyral to test the influence of insertion depth

def spyral(input, fs, electrodes, n_carriers, spread,depth=0 ,**kwargs):
    """Spyral: vocoder that utilizes multiple sinusoidal carriers to simulate current spread

        Parameters
        ----------
        input : array
            The input signal
        fs : scalar
            The sampling frequency
        electrodes : scalar or array
            If type==scalar, it represents the number of electrodes and each electrode will be 
            linearely distributed on an ERB scale between analysis_lo and analysis_hi.

            If type==array, each element represents the corresponding best frequency of each 
            electrode, and the number of electrodes is inferred from its length. Among other 
            things, this can be used to simulate warping, in which the cfs of analysis bands 
            may be different than the electrode positions.

        n_carriers : scalar
            Number of tone carriers
        spread : scalar
            Current spread [in -dB/Oct (negative!!)]. 
            Typical (Oxenham & Kreft 2014 + Nelson etal 2011) = -8 dB/octave.

        Kwargs
        ------
        analysis_lo : scalar 
            Lower bound of analysis filters, in Hz [default = 120 (Friesen et al.,2001)]
        analysis_hi : scalar 
            Upper bound of analysis filters, in Hz [default = 8658]
        carrier_lo : scalar 
            Lower bound of carriers, in Hz [default = 20]
        carrier_hi : scalar 
            Higher bound of carriers, in Hz [default = 20,000]
        analysis_cutoffs : array
            An array of cutoff frequencies to use. analysis_hi and lo are ignored. Must
            be one more than the number of electrodes.
        filt_env : scalar 
            Envelope filter cutoff, in Hz [default = 50]

        Returns
        -------
        out : array
            Vocoded input

        Example
        -------
        >>> out = spyral(signal, 44100, 20, 80, -8)

    """
    analysis_lo = kwargs.get('analysis_lo', 120) 
    analysis_hi = kwargs.get('analysis_hi', 8658)
    analysis_cutoffs = kwargs.get('analysis_cutoffs', None)
    carrier_lo = kwargs.get('carrier_lo', 20) 
    carrier_hi = kwargs.get('carrier_hi', 20000) 
    filt_env = kwargs.get('filt_env', 50)
    in_phase = kwargs.get('in_phase', False)
    fs = np.float32(fs)
    carrier_weights = kwargs.get('carrier_weights', np.ones(n_carriers, dtype=int)) #每个carrier的健康百分比

    rms_in = np.sqrt(np.mean(np.power(input, 2)))
    lp_filter = make_fir_filter(0, filt_env, fs)       # generate low-pass filter,  default 50Hz
    if np.isscalar(electrodes):
        cfs = np.array(generate_cfs(analysis_lo, analysis_hi, electrodes))     # electrodes' centre frequencies
    else:
        cfs = np.array(electrodes) # If not scalar, assume a list of cfs
    carrier_fs = generate_cfs(carrier_lo, carrier_hi, n_carriers) # tone carrier frequencies
    t = np.arange(0, len(input) / fs, 1 / fs)
    t_carrier = np.zeros((n_carriers, len(input)))
    if analysis_cutoffs is not None:
        ip_bands = analysis_cutoffs # User specified cutoffs
    else:
        ip_bands = np.array(generate_bands(analysis_lo, analysis_hi, cfs.size)) # lower/upper limits of each analysis band
    ip_bank = np.zeros((cfs.size, 512))
    envelope = np.zeros((cfs.size, len(input)))       # envelopes extracted per electrode
    mixed_envelope = np.zeros((n_carriers, len(input)))   # mixed envelopes to modulate carriers

    # Envelope extraction
    for j in range(cfs.size):
        ip_bank[j, :] = make_fir_filter(ip_bands[j, 0], ip_bands[j, 1], fs)   # analysis filterbank
        speechband = np.convolve(input, ip_bank[j, :], mode='same')
        envelope[j, :] = np.convolve(np.maximum(speechband,0), lp_filter, mode='same') # low-pass filter envelope

    ##########################################################################
    #adjust the depth in this step
    ##########################################################################
    adj_cfs=depth_adjust(cfs,depth)

    # weights applied to power envelopes
    for i in range(n_carriers):
        for j in range(cfs.size):
            mixed_envelope[i, :] += 10. ** (spread / 10. * np.abs(np.log2(adj_cfs[j] / carrier_fs[i]))) * envelope[j, :] ** 2.

    # sqrt to get back to amplitudes
    mixed_envelope = np.sqrt(mixed_envelope)
    out = np.zeros(len(input))

    if in_phase:
        phases = np.zeros(n_carriers)
    else:
        phases = np.random.rand(n_carriers) * 2 * np.pi

    # Generate carriers, modulate; randomise tone phases (particularly important for binaural!)
    for i in range(n_carriers):
#        t_carrier[i, :] = np.sin(2 * np.pi * (carrier_fs[i] * t + np.random.rand()))
        t_carrier[i, :] = np.sin(phases[i] + (2. * np.pi * carrier_fs[i] * t))*carrier_weights[i]

        out += mixed_envelope[i, :] * t_carrier[i, :]             # modulate carriers with mixed envelopes
#    out = out * 0.05 * np.sqrt(len(out)) / np.linalg.norm(out)    # rms scaled, to avoid saturation
    return out * (np.sqrt(np.mean(np.square(input))) / np.sqrt(np.mean(np.square(out))))

#start and end frequency for the CI
begin_fc=120
end_fc=8658

#depths adjustion for the insertion depth, in mm
depths=[-2,-1.5,-1,-0.5,0,1,2,2.5,3,3.5]

electroids_num=12

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
    loss.backward()
    return loss

def depth_adjust(cfs,depth=0):
    #move cfs to simulate different inplanting depth in insertion
    _depths=np.array([depth]*electroids_num)
    adjusted_depth=(35 / 2.1) * np.log10((np.array(cfs)/ 165.4) + 0.88)+_depths
    return 165.4 * (10**(2.1 * adjusted_depth / 35) - 0.88)

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
    librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    sig_ls=[]
    sentenses=[]
    for i in range(0,70):
        (sig,fs,enc) = pysndfile.sndio.read(librispeech_samples_ds[i]["file"])
        sig_ls.append(sig)
        sentenses.append(predict(sig,fs))
    return sentenses, sig_ls, fs

# load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

#read sound data
(sentenses,sig_ls, fs)=read_sound()

cfs=generate_cfs(begin_fc, end_fc, electroids_num)
record=[[cfs,3374.5793]]
f=open("record7.txt","w")

start = time.perf_counter() # timer
     
    
cfs=generate_cfs(begin_fc, end_fc, electroids_num)
record=[]
for depth in depths:
    loss=0
    bands=cfs_to_bands(cfs)
    for i in range(0,70):
        try:
            sig_out = spyral(sig_ls[i], fs, cfs, 80, -40, analysis_cutoffs=np.array(bands),depth=depth)
            loss += get_loss(sentenses[i],sig_out,fs)
        except:
            sig_out = 0
            loss+=60
            #break
    record.append([depth,loss])
    f.write(str(depth))
    f.write("loss: "+str(loss)+"\n")
    print(str(depth))
    print("loss: "+str(loss))


end = time.perf_counter()   
elapsed = end - start  
print("run time:", elapsed/60)
f.close()