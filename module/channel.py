import numpy as np

def AWGN(in_data,amp):
    '''
    Apply AWGN to the input signal
    
    Parameters
    -----------
    in_data: ndarray(n) 
        16PSK modulated signal

    Returns
    -----------
    out_data: ndarray(n)
        noised 16PSK modulated signal
    
    By SCUT Zening Lin, on 2021-01-06
    '''

    length = len(in_data)
    noise = amp*np.random.randn(length)
    data_n = in_data + noise

    return data_n