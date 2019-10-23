#! /usr/bin/python

import ann

traindata = ann.read_dataset('train.list')
trainlabel = ann.read_label('trainlabel.list')

hidden = 100
nn = ann.create_neural_net(960, hidden, 20)

testdata = ann.read_dataset('test1.list')
testlabel = ann.read_label('testlabel1.list')

epoch = 1000
learningrate = 0.2
momentum = 0.75

ann.train(nn, traindata, trainlabel, testdata, testlabel, epoch, learningrate, momentum, 1, 'classified.txt', 'classified2.txt')
