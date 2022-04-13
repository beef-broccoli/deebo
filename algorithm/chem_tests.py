import itertools

if __name__ == '__main__':
    com1 = ['a', 'b', 'c']
    com2 = ['d', 'e', 'f']

    l = list(itertools.product(com1, com2))



    print(l)

