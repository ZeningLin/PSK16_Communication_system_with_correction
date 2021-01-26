import numpy as np
import scipy
import scipy.io.wavfile
import wave
import matplotlib.pyplot as plt

import module.pcm as pcm
import module.psk16 as psk16
import module.channel as channel
import module.audio_func as audio_func

if __name__ == "__main__":
    # 文件读取并截取少量数据点
    fs,data_raw = scipy.io.wavfile.read('audio.wav')
    data_raw = data_raw[50000:50005]
    
    # PCM编码
    data_encoded_pcm,m = pcm.PCM_encode(data_raw)

    # 16PSK调制
    data_encoded_analog = psk16.psk16_modulate(data_encoded_pcm,10*fs,fs,50)

    # 加噪声
    data_encoded_analog_n = channel.AWGN(data_encoded_analog,0.3)
    
    # 16PSK相关解调
    data_decoded_pcm_correlated = psk16.psk16_correlated_demodulate(data_encoded_analog_n,10*fs,fs,50)
    
    # 16PSK相干解调
    data_decoded_pcm_coherent = psk16.psk16_coherent_demodulate(data_encoded_analog_n,10*fs,fs,50)
    
    # 解调后信号进行PCM译码
    data_decoded_raw_correlated = pcm.PCM_decode(data_decoded_pcm_correlated, m)
    data_decoded_raw_coherent = pcm.PCM_decode(data_decoded_pcm_coherent, m)
    
    # 绘制输入和输出系统的模拟信号波形
    plt.figure('original signal')
    plt.subplot(3,1,1)
    plt.title('raw signal')
    plt.plot(data_raw)
    plt.subplot(3,1,2)
    plt.title('final signal correlated')
    plt.plot(data_decoded_raw_correlated)
    plt.subplot(3,1,3)
    plt.title('final signal coherent')
    plt.plot(data_decoded_raw_coherent)
    
    # 绘制PCM编码信号
    plt.figure('PCM encoded signal')
    plt.subplot(3,1,1)
    plt.title('PCM encoded signal')
    plt.stem(data_encoded_pcm)
    plt.subplot(3,1,2)
    plt.title('PCM encoded signal correlated')
    plt.stem(data_decoded_pcm_correlated)
    plt.subplot(3,1,3)
    plt.title('PCM encoded signal coherent')
    plt.stem(data_decoded_pcm_coherent)

    # 绘制16PSK调制后的模拟波形
    plt.figure('16PSK modulated signal')
    plt.title('16PSK modulated signal')
    plt.plot(data_encoded_analog)

    plt.show()