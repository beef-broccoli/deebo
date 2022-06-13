
# make sure number is not 0 or 1.
def zero_nor_one(num):
    if num>1 or num<0:
        return num
    else:
        return min(max(float(num), 1e-7), 1-1e-7)

if __name__ == '__main__':
    print(zero_nor_one(0.9))