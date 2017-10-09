import random,sys
from os import listdir
#for each data file, such as manban_all.txt, divide it into train and test data sets with a 90-10 split

def count_num_lines(file_name):
    f=open(file_name,'r').read()
    return len([i for i in f if i=='\n'])




def split(file_name):
    num_lines=count_num_lines(file_name)
    test_size=num_lines/10
    training_size=num_lines-test_size
    print 'test,training:',test_size,training_size

    lines=open(file_name,'r').read().split('\n')
    lines=[i for i in lines if i!='']
    test_data=random.sample(lines,test_size)
    training_data=[i for i in lines if i not in test_data]

    return test_data,training_data



def main():
    path=sys.argv[1]
    onlyfiles=[i for i in listdir(path) if i.endswith('.txt')]
    for file_name in onlyfiles:
        fn = path + file_name
        output_test_name = fn.replace('.txt','_test.txt')
        output_train_name = fn.replace('.txt','_train.txt')
        test, train = split(fn)
        open(output_test_name,'w').close()
        output_test = open(output_test_name,'a')
        for i in test:
            output_test.write(i + '\n')
        output_test.close()

        open(output_train_name,'w').close()
        output_train = open(output_train_name,'a')
        for i in train:
            output_train.write(i + '\n')
        output_train.close()



if __name__ == '__main__':
    main()





