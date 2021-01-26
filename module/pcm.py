import numpy as np


def PCM_encode(in_data,vp=0):
    '''
    Apply A-law PCM encoding to the input signal
    
    Parameters
    -----------
    in_data: ndarray(n) 
        16PSK modulated signal
    
    vp: float
        maximum amplitute of the input signal

    Returns
    -----------
    out_data: ndarray(n*8)
        A-law PCM encoded signal
    
    Author
    ----------
    By SCUT Zening Lin, on 2021-01-06
    '''
    if vp==0:
        data_max = np.max(np.abs(in_data))
    else:
        data_max = vp

    in_data = in_data/data_max*4096

    dlength = len(in_data)
    out_data = np.zeros((dlength, 8))
    for i in range(dlength):
        # 极性码
        if in_data[i] > 0:
            out_data[i, 0] = 1
        else:
            out_data[i, 0] = 0 
        # 段落码
        if abs(in_data[i]) < 32:
            out_data[i, 1], out_data[i, 2], out_data[i, 3], step, st = 0, 0, 0, 2, 0
        elif abs(in_data[i]) < 64:
            out_data[i, 1], out_data[i, 2], out_data[i, 3], step, st = 0, 0, 1, 2, 32
        elif abs(in_data[i]) < 128:
            out_data[i, 1], out_data[i, 2], out_data[i, 3], step, st = 0, 1, 0, 4, 64
        elif abs(in_data[i]) < 256:
            out_data[i, 1], out_data[i, 2], out_data[i, 3], step, st = 0, 1, 1, 8, 128
        elif abs(in_data[i]) < 512:
            out_data[i, 1], out_data[i, 2], out_data[i, 3], step, st = 1, 0, 0, 16, 256
        elif abs(in_data[i]) < 1024:
            out_data[i, 1], out_data[i, 2], out_data[i, 3], step, st = 1, 0, 1, 32, 512
        elif abs(in_data[i]) < 2048:
            out_data[i, 1], out_data[i, 2], out_data[i, 3], step, st = 1, 1, 0, 64, 1024
        else:
            out_data[i, 1], out_data[i, 2], out_data[i, 3], step, st = 1, 1, 1, 128, 2048
        # 段内码
        seg_index = (int((abs(in_data[i]) - st) / step))
        if seg_index < 1:
            out_data[i,4:]=np.array([0,0,0,0])
        elif seg_index < 2:
            out_data[i,4:]=np.array([0,0,0,1])
        elif seg_index < 3:
            out_data[i,4:]=np.array([0,0,1,0])
        elif seg_index < 4:
            out_data[i,4:]=np.array([0,0,1,1])
        elif seg_index < 5:
            out_data[i,4:]=np.array([0,1,0,0])
        elif seg_index < 6:
            out_data[i,4:]=np.array([0,1,0,1])
        elif seg_index < 7:
            out_data[i,4:]=np.array([0,1,1,0])
        elif seg_index < 8:
            out_data[i,4:]=np.array([0,1,1,1])
        elif seg_index < 9:
            out_data[i,4:]=np.array([1,0,0,0])
        elif seg_index < 10:
            out_data[i,4:]=np.array([1,0,0,1])
        elif seg_index < 11:
            out_data[i,4:]=np.array([1,0,1,0])
        elif seg_index < 12:
            out_data[i,4:]=np.array([1,0,1,1])
        elif seg_index < 13:
            out_data[i,4:]=np.array([1,1,0,0])
        elif seg_index < 14:
            out_data[i,4:]=np.array([1,1,0,1])
        elif seg_index < 15:
            out_data[i,4:]=np.array([1,1,1,0])
        else:
            out_data[i,4:]=np.array([1,1,1,1])
 
    return out_data.reshape(8 * dlength), data_max


def PCM_decode(in_data, v):
    '''
    Apply A-law PCM decoding to the input signal
    
    Parameters
    -----------
    in_data: ndarray(n) 
        A-law encoded signal
     
    v: int
        maximum voltage of the decoded signal
   
    Returns
    -----------
    out_data: ndarray(n/8)
        A-law decoded signal
    
    Author
    -----------
    By SCUT Zening Lin, on 2021-01-06, 
    
    Based on the project `python_sound_open` by Jie Y.
    '''
    length = len(in_data)//8
    in_data = in_data.reshape(length, 8)
    slot = np.array([0, 32, 64, 128, 256, 512, 1024, 2048])
    step = np.array([2, 2, 4, 8, 16, 32, 64, 128])
    out_data = np.zeros(length)
    for i in range(length):
        # 符号位转换为符号
        sgn = 2 * in_data[i, 0] - 1
        # 段落码转换为段落初始电平
        seg_index = int(in_data[i, 1] * 4 + in_data[i, 2] * 2 + in_data[i, 3])
        st = slot[seg_index]
        dt = (in_data[i, 4] * 8 + in_data[i, 5] * 4 + in_data[i, 6] * 2 + in_data[i, 7]) * step[seg_index]
        out_data[i] = sgn * (st + dt + 0.5 * step[seg_index]) / 4096 *v
        # out_data[i] = out_data[i]*(v-s)+s
    return out_data

if __name__ == "__main__":
    # 测试数据： -2V,2V,-0.74V,0V
    data_raw = np.array([169646,1352,321,-0.74,1,0,-2])

    data_encoded,m = PCM_encode(data_raw)
    data_decoded = PCM_decode(data_encoded,m)

    print(data_raw,'\n',data_encoded,'\n',data_decoded)