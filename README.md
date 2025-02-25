import random

def exp(x, terms=10):
        result = 1
        factorial = 1
        power = 1
        for i in range(1, terms):
            factorial *= i
            power *=x
            result += power/factorial
        return result
    
def tanh(x):
    e_pos = exp(x)
    e_neg = exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)

def forward_propagation(i1, i2, weights, biases):
    
    h1 = tanh(i1 * weights['w1'] + i2 * weights['w2'] + biases['b1'])
    h2 = tanh(i1 * weights['w3'] + i2 * weights['w4'] + biases['b1'])
    
    o1 = tanh(i1 * weights['w5'] + i2 * weights['w6'] + biases['b2'])
    o2 = tanh(i1 * weights['w7'] + i2 * weights['w8'] + biases['b2'])
    
    
    return o1, o2

weights = {key: random.uniform(-0.5, 0.5) for key in ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8']}
biases = {'b1': 0.5, 'b2': 0.7}

#input values
i1, i2 = 0.05, 0.10

o1, o2 = forward_propagation(i1, i2, weights, biases)
    
    
#print results
print("o1:", o1)
print("o2:", o2)

