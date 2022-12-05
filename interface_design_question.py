def missingchar(s, t):
    sd = {}
    td = {}
    for i in s:
        if i in sd.keys():
            sd[i] += 1
        else:
            sd[i] = 1
    for i in t:
        if i in td.keys():
            td[i] += 1
        else:
            td[i] = 1
    for k in sd:
        if sd[k] != td[k]:
            return k
    return 'Nothing'

# t = 'abbeeras'
# s = 'aserabb'
t = 'abbas'
s = 'saba'
t = 'aa'
s = 'a'

print(missingchar(s, t))