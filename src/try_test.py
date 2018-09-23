
predicted = ['hello', 'people', 'henjdf', 'girl','_eos_']

original = ['hello', 'people', 'henjdf','_eos_','_eos_']

print(original.count('_eos_'))
counterr=0
for value in predicted:
    if value in original and  value != '_eos_':
        counterr = counterr +1

print(counterr)
