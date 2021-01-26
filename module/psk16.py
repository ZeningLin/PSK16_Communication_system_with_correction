import numpy as np
import scipy
import scipy.signal
import math


def psk16_modulate(in_data,fc,fs,N):
    """
    Apply 16PSK modulation to the input signal

    Parameters
    -----------
    in_data: ndarray(n) 
        A-law encoded signal
     
    fc: int
        carrier frequency 
    
    fs: int
        symbol rate
     
    N: int
        data per analog period

    Returns
    -----------
    out_data: ndarray(n*N*fc/(fs*4))
        modulated signal
    
    Author
    ----------
    By SCUT Zening Lin, Yushu Li, Qiheng Tan, Qingfeng Wang on 2020-01-06
    """
    length = len(in_data)//4
    in_data = in_data.reshape(length,4)
    ai_data = np.zeros((length,2))
     
    ratio = int(fc/fs)
    out_data =np.zeros(length*N*ratio)

    carrier=np.zeros((N,2))
    for i in range(N):
        carrier[i] = [math.cos(2*math.pi/(N-1)*i),-math.sin(2*math.pi/(N-1)*i)]

    phasetab = np.array(
        [[0.9807852804032304,0.19509032201612825],
        [0.8314696123025452,0.5555702330196022],
        [0.5555702330196023,0.8314696123025452],
        [0.19509032201612833,0.9807852804032304],
        [-0.1950903220161282,0.9807852804032304],
        [-0.555570233019602,0.8314696123025453],
        [-0.8314696123025453,0.5555702330196022],
        [-0.9807852804032304,0.1950903220161286],
        [-0.9807852804032304,-0.19509032201612836],
        [-0.8314696123025455,-0.555570233019602],
        [-0.5555702330196022,-0.8314696123025452],
        [-0.19509032201612866,-0.9807852804032303],
        [0.1950903220161283,-0.9807852804032304],
        [0.5555702330196026,-0.831469612302545],
        [0.8314696123025452,-0.5555702330196022],
        [0.9807852804032303,-0.19509032201612872]]
    )

    for i in range(length):
        dec_data = in_data[i,0]*8+in_data[i,1]*4+in_data[i,2]*2+in_data[i,3]
        ai_data[i] = phasetab[int(dec_data)]
        a,b = ai_data[i][0],ai_data[i][1]
        for j in range(N):
            for k in range(ratio):
                out_data[i*N*ratio+j+k*N]=a*carrier[j,0]+b*carrier[j,1]

    return out_data

def psk16_correlated_demodulate(in_data,fc,fs,N):
    """
    Apply 16PSK correlated demodulation to the input signal

    Parameters
    -----------
    in_data: ndarray(n) 
        16PSK modulated signal
     
    fc: int
        carrier frequency 
    
    fs: int
        symbol rate
     
    N: int
        data per analog period

    Returns
    -----------
    out_data: ndarray(n*N*fc/(fs*4))
        modulated signal
    
    Author
    ----------
    By SCUT Zening Lin, Yushu Li, Qiheng Tan, Qingfeng Wang on 2020-01-06
    """
    ratio = int(fc/fs)
    length = len(in_data)//(ratio*N)
    out_data = np.zeros((length,4))

    # 相关解调本地载波信号的生成
    carrier_n = np.zeros((N,2))
    for i in range(N):
        carrier_n[i] = [math.cos(2*math.pi/(N-1)*i),-math.sin(2*math.pi/(N-1)*i)]

    # 16PSK星座图角度表
    angletab = np.zeros(16)
    for i in range(16):
        angletab[i] = math.pi*i/8+math.pi/16

    # 角度值的十进制转换为二进制编码输出
    dec2bin = np.array(
        [[0,0,0,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,0,1,1],
        [0,1,0,0],
        [0,1,0,1],
        [0,1,1,0],
        [0,1,1,1],
        [1,0,0,0],
        [1,0,0,1],
        [1,0,1,0],
        [1,0,1,1],
        [1,1,0,0],
        [1,1,0,1],
        [1,1,1,0],
        [1,1,1,1]]
    )
    
    for i in range(length):
        r1 = 0
        r2 = 0

        # 基信号与接收信号相乘并积分
        for j in range(ratio):
            for k in range(N): 
                r1 += carrier_n[k][0] * in_data[i*ratio*N+j*N+k]
                r2 += carrier_n[k][1] * in_data[i*ratio*N+j*N+k]       
        
        # 判决，利用解调结果对应的角度值与星座图比较
        # 接收信号的角度计算
        angle_received = math.atan(r2/r1)
        if r1 < 0:
            angle_received += math.pi
        elif r2 < 0:
            angle_received += 2*math.pi
        # 比较接收信号与星座图的角度值得到判决结果
        code_dec = 0
        for k in range(16):
            if (angletab[k]- math.pi/16 <= angle_received) and (angle_received <= angletab[k]+math.pi/16):
                code_dec = k
                break
        code_bin = dec2bin[code_dec]
        out_data[i] = code_bin

    return out_data.reshape(length*4)


def psk16_coherent_demodulate(in_data,fc,fs,N):
    """
    Apply 16PSK coherent demodulation to the input signal

    Parameters
    -----------
    in_data: ndarray(n) 
        16PSK modulated signal
     
    fc: int
        carrier frequency 
    
    fs: int
        symbol rate
     
    N: int
        data per analog period

    Returns
    -----------
    out_data: ndarray(n*N*fc/(fs*4))
        modulated signal
    
    By SCUT Zening Lin, Yushu Li, Qiheng Tan, Qingfeng Wang on 2020-01-07
    """
    ratio = int(fc/fs)
    length = len(in_data)//(ratio*N)
    out_data = np.zeros((length,4))

    # 相干解调本地载波信号的生成
    carrier_n = np.zeros((N,2))
    for i in range(N):
        carrier_n[i] = [math.cos(2*math.pi/(N-1)*i),-math.sin(2*math.pi/(N-1)*i)]

    # 16PSK星座图角度表
    angletab = np.zeros(16)
    for i in range(16):
        angletab[i] = math.pi*i/8+math.pi/16

    # 角度值的十进制转换为二进制编码输出
    dec2bin = np.array(
        [[0,0,0,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,0,1,1],
        [0,1,0,0],
        [0,1,0,1],
        [0,1,1,0],
        [0,1,1,1],
        [1,0,0,0],
        [1,0,0,1],
        [1,0,1,0],
        [1,0,1,1],
        [1,1,0,0],
        [1,1,0,1],
        [1,1,1,0],
        [1,1,1,1]]
    )

    in_data_c = np.zeros((length,N))  # 存放与cos相乘后的数据
    in_data_s = np.zeros((length,N))  # 存放与-sin相乘后的数据
    F_in_data_c = np.zeros((length,512))  # 存放与cos相乘后的频谱数据
    F_in_data_s = np.zeros((length,512))  # 存放与-sin相乘后的频谱数据
    # 采用8阶IIR巴特沃斯滤波器
    LP_filter_b, LP_filter_a = scipy.signal.butter(8, 1/N)

    # 基信号与接收信号相乘    
    for i in range(length):
        for j in range(ratio):
            for k in range(N): 
                in_data_c[i][k] += (carrier_n[k][0] * in_data[i*ratio*N+j*N+k])/ratio
                in_data_s[i][k] += (carrier_n[k][1] * in_data[i*ratio*N+j*N+k])/ratio
    
    
        # 伪低通滤波操作,对正弦信号取均值即能得到信号的直流分量
        r1=np.mean(in_data_c[i])
        r2=np.mean(in_data_s[i])
        
        # 判决，利用解调结果对应的角度值与星座图比较
        # 接收信号的角度计算
        angle_received = math.atan(r2/r1)
        if r1 < 0:
            angle_received += math.pi
        elif r2 < 0:
            angle_received += 2*math.pi
        # 比较接收信号与星座图的角度值得到判决结果
        code_dec = 0
        for k in range(16):
            if (angletab[k]- math.pi/16 <= angle_received) and (angle_received <= angletab[k]+math.pi/16):
                code_dec = k
                break
        code_bin = dec2bin[code_dec]
        out_data[i] = code_bin

    return out_data.reshape(length*4)



if __name__ == "__main__":
    # 测试数据：所有4位二进制数
    test_array = np.array([0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,1,1,1,1,0,0,0,1,0,0,1,1,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,1,1,0,1,1,1,1])
    # 经16PSK调制后的信号
    a = psk16_modulate(test_array,200,50,500)
    # 经16PSK相关解调后的信号
    b = psk16_correlated_demodulate(a,200,50,500).reshape(len(test_array))
    # 经16PSK相干解调后的信号
    c = psk16_coherent_demodulate(a,200,50,500).reshape(len(test_array))

    hamming_dis_correlated = 0
    hamming_dis_coherent = 0
    for i in range(len(test_array)):
        if test_array[i]!=b[i]:
            hamming_dis_correlated+=1
        if test_array[i]!=c[i]:
            hamming_dis_coherent+=1
    print ('error rate of correlated demodulation:',hamming_dis_coherent/len(test_array),'\n',
    'error rate of coherent demodulation:',hamming_dis_correlated/len(test_array))