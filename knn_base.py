#!/usr/bin/env python2
#coding:utf-8
'basic KNN algorithm'
__author__='yongjia sang'


from numpy import *
import operator

def readDataFromFile(filename):
    file_handler = open(filename,'r')
    number_of_rows = len(file_handler.readlines())
    training_data_mat = zeros((number_of_rows, 3))
    class_label_vector = []
    index = 0
    file_handler.close()
    file_handler = open(filename,'r')
    for line in file_handler:
        elements = line.split('\t')
        training_data_mat[index, :] = elements[0:3]
        class_label_vector.append(int(elements[-1]))
        #print class_label_vector
        index += 1 
        line = line.strip()
    file_handler.close()
    return training_data_mat, class_label_vector

def normalizeData(training_data_mat):
    min_values = training_data_mat.min(0)
    max_values = training_data_mat.max(0)
    ranges = max_values - min_values
    norm_training_data_mat = zeros(shape(training_data_mat))
    m = training_data_mat.shape[0]
    norm_training_data_mat = training_data_mat - tile(min_values, (m,1))
    norm_training_data_mat = norm_training_data_mat / tile(ranges,(m,1))
    return norm_training_data_mat, ranges, min_values

def classifier(input_data, norm_training_data_mat, class_label_vector, k):
    number_of_rows = norm_training_data_mat.shape[0]
    diff_mat = tile(input_data,(number_of_rows,1)) - norm_training_data_mat
    square_diff_mat = diff_mat**2
    square_distance = square_diff_mat.sum(axis=1)
    distances = square_distance**0.5
    sorted_distances = distances.argsort()
    class_vote = {}
    for i in range(k):
        #label = 1
        label = class_label_vector[sorted_distances[i]]
        class_vote[label] = class_vote.get(label,0)+1
        sorted_class_vote = sorted(class_vote.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_vote[0][0]
    pass
def runTestData():
    ratio = 0.1
    training_data_mat, class_label_vector = readDataFromFile('/data1/yongjia/machineLearning/KNN/trainingAndTestData/datingTestSet2.txt')
    norm_training_data_mat, ranges, min_values = normalizeData(training_data_mat)
    m = norm_training_data_mat.shape[0]
    test_data_number = int(m*ratio)
    error_count = 0
    #for i in range(1):
    for i in range(test_data_number):
        result = classifier(norm_training_data_mat[i,:],norm_training_data_mat[test_data_number:m,:],class_label_vector[test_data_number:m],9)
        print "the classifier result is: %d, the real answer is: %d" %(result,class_label_vector[i])
        if(result != class_label_vector[i]):
            error_count +=1
    print "the error rate is: %f" %(error_count/float(test_data_number))
    pass
if __name__ == '__main__':
    #readDataFromFile('/data1/yongjia/machineLearning/KNN/trainingAndTestData/datingTestSet2.txt')
    runTestData()
