import numpy as np

def correction_en(in_data):
    """
    Apply (12,8) correction encoding to the input signal

    Parameters
    -----------
    in_data: ndarray(n) 
        correction encoded signal

    Returns
    -----------
    out_data: ndarray(n)
        correction encoded signal
    
    Author
    ----------
    By SCUT Zening Lin, Yushu Li, Qiheng Tan, Qingfeng Wang on 2020-01-06
    """
    generate = np.array(
        [[1,0,0,0,0,0,0,0,1,1,1,0],
        [0,1,0,0,0,0,0,0,1,0,0,1],
        [0,0,1,0,0,0,0,0,0,1,0,1],
        [0,0,0,1,0,0,0,0,1,1,0,1],
        [0,0,0,0,1,0,0,0,0,0,1,1],
        [0,0,0,0,0,1,0,0,1,0,1,1],
        [0,0,0,0,0,0,1,0,0,1,1,1],
        [0,0,0,0,0,0,0,1,1,1,1,1]]
    )


    length = len(in_data)//8
    in_data = in_data.reshape(length,8)
    in_data_grey = np.zeros((length,8))
    out_data = np.zeros((length,12))

    # 格雷码编码
    for i in range(length):
        for j in range(8):
            if j==0:
                in_data_grey[i][j] = in_data[i][j]
            elif in_data[i][j]==in_data[i][j-1]:
                in_data_grey[i][j] = 0
            else:
                in_data_grey[i][j] = 1

        out_data[i] = np.matmul(in_data_grey[i],generate)
    
    out_data = out_data.reshape(length*12)
    
    # 模二运算修正
    for i in range(length*12):
        if out_data[i]%2 == 0:
            out_data[i] = 0
        else:
            out_data[i] = 1

    return out_data



def correction_de(in_data):
    """
    Apply (12,8) correction decoding to the input signal

    Parameters
    -----------
    in_data: ndarray(n) 
        correction encoded signal

    Returns
    -----------
    out_data: ndarray(n)
        correction decoded signal
    
    Author
    ----------
    By SCUT Zening Lin, Yushu Li, Qiheng Tan, Qingfeng Wang on 2020-01-06
    """

    supervision = np.array(
        [[1,1,0,1,0,1,0,1,1,0,0,0],
        [1,0,1,1,0,0,1,1,0,1,0,0],
        [1,0,0,0,1,1,1,1,0,0,1,0],
        [0,1,1,1,1,1,1,1,0,0,0,1]]
    )

    length = len(in_data)//12
    in_data_grey = in_data.reshape(length,12)
    syndrome = np.zeros(4)
    out_data_bin = np.zeros((length,8))


    # 格雷码译码
    for i in range(length):
        error_pattern = np.zeros(12)
        syndrome = np.transpose(np.matmul(supervision, np.transpose(in_data_grey[i])))
        for j in range(4):
            if syndrome[j]%2 == 0:
                syndrome[j] = 0
            else:
                syndrome[j] = 1

        syndrome = syndrome.tolist()
        if syndrome == [0,0,0,1]:
            error_pattern = np.array([0,0,0,0,0,0,0,0,0,0,0,1])
        elif syndrome == [0,0,1,0]:
            error_pattern = np.array([0,0,0,0,0,0,0,0,0,0,1,0])
        elif syndrome == [0,1,0,0]:
            error_pattern = np.array([0,0,0,0,0,0,0,0,0,1,0,0])
        elif syndrome == [1,0,0,0]:
            error_pattern = np.array([0,0,0,0,0,0,0,0,1,0,0,0])
        elif syndrome == [1,1,1,1]:
            error_pattern = np.array([0,0,0,0,0,0,0,1,0,0,0,0])
        elif syndrome == [0,1,1,1]:
            error_pattern = np.array([0,0,0,0,0,0,1,0,0,0,0,0])
        elif syndrome == [1,0,1,1]:
            error_pattern = np.array([0,0,0,0,0,1,0,0,0,0,0,0])
        elif syndrome == [0,0,1,1]:
            error_pattern = np.array([0,0,0,0,1,0,0,0,0,0,0,0])
        elif syndrome == [1,1,0,1]:
            error_pattern = np.array([0,0,0,1,0,0,0,0,0,0,0,0])
        elif syndrome == [0,1,0,1]:
            error_pattern = np.array([0,0,1,0,0,0,0,0,0,0,0,0])
        elif syndrome == [1,0,0,1]:
            error_pattern = np.array([0,1,0,0,0,0,0,0,0,0,0,0])
        elif syndrome == [1,1,1,0]:
            error_pattern = np.array([1,0,0,0,0,0,0,0,0,0,0,0])
        elif syndrome == [1,1,0,0]:
            error_pattern = np.array([0,0,0,0,0,0,0,0,1,1,0,0])
        elif syndrome == [1,0,1,0]:
            error_pattern = np.array([0,0,0,0,0,0,0,0,1,0,1,0])
        elif syndrome == [0,1,1,0]:
            error_pattern = np.array([0,0,0,0,0,0,0,0,0,1,1,0])

        in_data_grey[i] += error_pattern
        for j in range(12):
            if in_data_grey[i][j]%2 == 0:
                in_data_grey[i][j] = 0
            else:
                in_data_grey[i][j] = 1

        # 格雷码译码
        for j in range(8):
            if j==0:
                out_data_bin[i][j] = in_data_grey[i][j]
            elif in_data_grey[i][j]==out_data_bin[i][j-1]:
                out_data_bin[i][j] = 0
            else:
                out_data_bin[i][j] = 1
    
    return out_data_bin.reshape(length*8)



if __name__ == "__main__":
    test = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1])

    a = correction_en(test)

    a[12]=1

    b = correction_de(a)

    print(b)