
# make sure number is not 0 or 1.
def zero_nor_one(num):
    if num>1 or num<0:
        return num
    else:
        return min(max(float(num), 1e-7), 1-1e-7)


def fill_list(li, desired_len, to_fill):
    if li is None:
        return [to_fill]*desired_len
    else:
        return li + [to_fill]*(desired_len-len(li))

if __name__ == '__main__':
    print(fill_list(None, 6, 9))