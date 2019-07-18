import random
import numpy as np

def __number_sampler(max_digit=3):
    digits = ['0','1','2','3','4','5','6','7','8','9']
    k = random.randint(1,max_digit)
    string = ''.join(random.sample(digits,k))
    if string[0] == '0':
        return __number_sampler(max_digit=max_digit)
    return string

def __operator_sampler():
    operators = ['+','-','/','*']
    return random.choice(operators)

def __get_random_sequence(n_input=100, max_len_limit=15, max_digit=3):
    ret = []
    for i in range(n_input):
        length = random.randint(1,max_len_limit)
        length = length if length%2==1 else length+1
        #length = max_len_limit # added for simplicity, later delete this line to make problem more complicated
        s = ''
        for j in range(length):
            if j % 2 == 0:
                s += __number_sampler(max_digit=max_digit)
            else:
                s += __operator_sampler()
        while len(s) < max_len_limit: s += ' ' ## for variant input
        ret.append(s)
    return ret

def __one_hot_encoding(alphabet, max_len):
    x = np.zeros((len(alphabet),len(alphabet)),dtype=np.int32)
    dict_ = {}
    for i in range(x.shape[0]):
        x[i,i] = 1
        dict_[alphabet[i]] = x[i]
    return dict_

def __create_one_hot_seq(alphabet, max_len, input_sequence):
    alphabet2encod = __one_hot_encoding(alphabet, max_len)
    ret = np.zeros((len(input_sequence),max_len,len(alphabet)))
    for j, inp in enumerate(input_sequence):
        temp = np.zeros((len(inp),len(alphabet)))
        for i in range(temp.shape[0]):
            temp[i,:] = alphabet2encod[inp[i]]
        ret[j,:,:] = temp
    return ret

def get_input(n_input=n_input, max_len_limit=max_len_limit, max_digit=1, alphabet):
    input_sequence = __get_random_sequence(n_input=n_input, max_len_limit=max_len, max_digit=1)
    one_hot_input = __create_one_hot_seq(alphabet, max_len, input_sequence)
    input_seq = torch.FloatTensor(one_hot_input).permute(1,0,2)
    
    target_values = []
    for i in range(len(input_sequence)):
        target_values.append(eval(input_sequence[i]))

    target_values = torch.FloatTensor(target_values).unsqueeze(1)
    target_values = target_values.to(device)
    
    return input_seq, target_values
