import numpy as np
import scipy
import scipy.io.wavfile
import wave
import matplotlib.pyplot as plt
import time

import module.pcm as pcm
import module.psk16 as psk16
import module.channel as channel
import module.audio_func as audio_func
import module.correction as correction

if __name__ == "__main__":
    time_start=time.time()

    fs,data_raw = scipy.io.wavfile.read('audio.wav')
    # PCM编码
    data_encoded_pcm,m = pcm.PCM_encode(data_raw)
    # 差错控制编码
    data_encoded_pcm_corr = correction.correction_en(data_encoded_pcm)
    # 16PSK调制，每载波周期10个点，每码元周期10个载波周期
    data_encoded_analog = psk16.psk16_modulate(data_encoded_pcm_corr,10*fs,fs,10)
    # 加噪声，噪声增益0.3
    data_encoded_analog_n = channel.AWGN(data_encoded_analog,0.3)
    
    
    # 对接收的信号进行相关解调
    data_decoded_pcm_correlated = psk16.psk16_correlated_demodulate(data_encoded_analog_n,10*fs,fs,10)
    # 差错控制译码
    data_encoded_pcm_correlated_corr = correction.correction_de(data_decoded_pcm_correlated)
    # PCM译码
    data_decoded_raw_correlated = pcm.PCM_decode(data_encoded_pcm_correlated_corr, m)
    
    
    # 对接收的信号进行相干解调
    # data_decoded_pcm_coherent = psk16.psk16_coherent_demodulate(data_encoded_analog_n,10*fs,fs,10)
    # 差错控制译码
    # data_encoded_pcm_coherent_corr = correction.correction_de(data_decoded_pcm_coherent)
    # PCM译码
    # data_decoded_raw_coherent = pcm.PCM_decode(data_encoded_pcm_coherent_corr, m)
    
    
    # 将解调译码后的数据转换为wav文件
    audio_func.audiowrite(data_decoded_raw_correlated,fs,'audio_correlated_decoded_4.wav')
    # audio_func.audiowrite(data_decoded_raw_coherent,fs,'audio_coherent_decoded_4_corr.wav')

    time_end=time.time()

    with open ('coherent_corr_pcm.txt','w') as f:
        err = 0
        length = len(data_encoded_pcm)//8
        for i in range(length):
            for j in range(8):
                if data_encoded_pcm[8*i+j] != data_encoded_pcm_correlated_corr[8*i+j]:
                    err+=1
                    break

        errate = err/length
        f.write(str(err))
        f.write('\n')
        f.write(str(errate))
        f.write('\n')
        f.write('time cost:')
        f.write(str(time_end-time_start))


    # 记录系统的运行时间和误码率_相干解调
    # with open ('coherent_corr_pcm.txt','w') as f:
    #     err = 0
    #     length = len(data_encoded_pcm)//8
    #     for i in range(length):
    #         for j in range(8):
    #             if data_encoded_pcm[8*i+j] != data_encoded_pcm_coherent_corr[8*i+j]:
    #                 err+=1
    #                 break

    #     errate = err/length
    #     f.write(str(err))
    #     f.write('\n')
    #     f.write(str(errate))
    #     f.write('\n')
    #     f.write('time cost:')
    #     f.write(str(time_end-time_start))