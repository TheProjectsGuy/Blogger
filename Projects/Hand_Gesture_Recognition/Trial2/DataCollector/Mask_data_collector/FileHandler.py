number = 0
try:
    with open('Index.txt', 'r+') as fin:
        number = int(fin.read())
        print(number)
except:
    with open('Index.txt', 'w+') as fout:
        fout.write('0')
else:
    with open('Index.txt', 'w+') as fout:
        fout.write(str(number + 1))
