import scipy
import scipy.io.wavfile
import wave
import numpy as np

import module.pcm as pcm

if __name__ == "__main__":
    fs1,data_raw = scipy.io.wavfile.read('audio.wav')
    fs2,data_correlated = scipy.io.wavfile.read('./result/audio_correlated_decoded.wav')
    # fs3,data_coherent = scipy.io.wavfile.read('./result/audio_coherent_decoded.wav')

    data_raw_pcm, m = pcm.PCM_encode(data_raw)
    data_correlated_pcm, mm1= pcm.PCM_encode(data_correlated,m)
    # data_coherent_pcm, mm2 = pcm.PCM_encode(data_coherent,m)

    # length = len(data_raw)
    # mean_raw = np.mean(data_raw)

    # err_correlated = 0
    # err_coherent = 0
    # for i in range(length):
    #         err_correlated+=abs(data_correlated[i]-data_raw[i])
    #         err_coherent+=abs(data_coherent[i]-data_raw[i])

    # errate_correlated = err_correlated/(mean_raw*length)
    # errate_coherent = err_coherent/(mean_raw*length)

    # with open ('coherent_pcm.txt','w') as f:
    #     err = 0
    #     length = len(data_raw_pcm)//8
    #     for i in range(length):
    #         for j in range(8):
    #             if data_raw_pcm[8*i+j] != data_coherent_pcm[8*i+j]:
    #                 err+=1
    #                 break

    #     errate = err/length
    #     f.write(str(err))
    #     f.write('\n')
    #     f.write(str(errate))
    #     f.write('\n')

    with open ('correlated_pcm.txt','w') as f:
        err = 0
        length = len(data_raw_pcm)//8
        for i in range(length):
            for j in range(8):
                if data_raw_pcm[8*i+j] != data_correlated_pcm[8*i+j]:
                    err+=1
                    break

        errate = err/length
        f.write(str(err))
        f.write('\n')
        f.write(str(errate))
        f.write('\n')