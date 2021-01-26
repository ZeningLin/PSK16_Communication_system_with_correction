import wave
import pyaudio
import numpy as np
import scipy

   
def audioplayer(path, frames_per_buffer=1024):
    '''
    播放语音文件
    
    :param frames_per_buffer:
    :return:
    
    2020-2-25   Jie Y.  Init
    '''
    wf = wave.open(path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(frames_per_buffer)
    while data != b'':
        stream.write(data)
        data = wf.readframes(frames_per_buffer)

    stream.stop_stream()
    stream.close()
    p.terminate()

def audiowrite(data, fs, path, binary=True, channel=1):
    '''
    信息写入到.wav文件中
    :param data: 语音信息数据
    :param fs: 采样率(Hz)
    :param binary: 是否写成二进制文件(只有在写成二进制文件才能用audioplayer播放)
    :param channel: 通道数
    :param path: 文件路径
    :return:
   
    2020-2-25   Jie Y.  Init

    Modified by SCUT Yushu Li on 2020-01-07
    '''

    data = data.astype(np.int16)

    if binary:
        wf = wave.open(path, 'wb')
        wf.setframerate(fs)
        wf.setnchannels(channel)
        wf.setsampwidth(2)
        wf.writeframes(b''.join(data))
    else:
        scipy.io.wavfile.write(path, fs, data)