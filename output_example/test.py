
import spyral
import pysndfile
import numpy as np

(standard,fs,enc) = pysndfile.sndio.read("/home/jyun/spyral/test/test_clean.wav")
(sig,fs,enc) = pysndfile.sndio.read("/home/jyun/spyral/test/test_clean.wav")


MCMC = np.array([140.59635654752566, 285.82416802791664, 559.4170092151569, 637.549021502949, 966.2858781864022, 1347.5752203809398, 1469.1339726849128, 2038.7003497803862, 2489.7548620535676, 3717.922699732674, 5337.2015231361265, 7900.912269404996])
ga=np.array([145.2808,  342.2637,  465.8460,  626.1932,  803.2255, 1209.8323, 1582.7736, 2156.8635, 2704.7981, 3970.2871, 5877.5977, 6593.0181])
electroids_num=12

def cfs_to_bands(cfs):
    m=np.array([168]*(electroids_num+1))
    cuts=list(np.log(cfs+m[1:]))
    cuts=np.array([2*cuts[0]-cuts[1]]+cuts+[2*cuts[-1]-cuts[-2]])
    block=np.exp([(cuts[i]+cuts[i+1])/2 for i in range(0,electroids_num+1)])-m
    bands=[[block[i],block[i+1]] for i in range(0,len(block)-1)]
    return np.array(bands)
bands=cfs_to_bands(MCMC)

out = spyral.spyral(sig, fs, ga, 80, -20, analysis_cutoffs=np.array(bands))

pysndfile.sndio.write('test/test_ga.wav',out,fs,format='wav')

out = spyral.spyral(sig, fs, MCMC, 80, -20, analysis_cutoffs=np.array(bands))

pysndfile.sndio.write('test/test_MCMC.wav',out,fs,format='wav')