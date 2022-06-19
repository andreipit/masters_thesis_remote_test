import time

def convert_dict_to_text(dictionary):
    pass


if __name__ == '__main__':
    x = {}
    x['var1'] = 11
    x['var2'] = 22
    print('start')
    n = 0
    while True:
        n += 1
        print(str(n), ' ', end='')


        time.sleep(0.1)

        x['var1'] += 1
        x['var2'] += 1

        filename: str = 'utils/test/test_txt_as_inspector.txt'
        #open(filename, 'w').close() # clear

        f = open(filename, "a")
        f.seek(0)
        f.truncate()
        for i in x:
            f.write(str(i) + '=' + str(x[i]) + '\n')
        f.close()


    #print('hi')
    ##with open('utils/test/test_txt_as_inspector.txt') as f:
    ##    lines = f.readlines()
    ##    print(lines)

    #f = open('utils/test/test_txt_as_inspector.txt', "a")
    #f.write("Now \n")
    #f.close()
    #while True:
    #    time.sleep(1)
    #    f = open('utils/test/test_txt_as_inspector.txt', "a")
    #    f.write("Now2 \n")
    #    f.close()

    ##while True:
    ##    with open('utils/test/test_txt_as_inspector.txt') as f:
    ##        lines = f.readlines()
    ##        print(lines)


