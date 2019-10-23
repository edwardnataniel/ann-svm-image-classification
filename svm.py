import os
from svmutil import *

out = open('train_data', 'w')
label = 0

print 'constructing training data...'

# construct training data
for root, dirs, fil in os.walk('.'):
	if root != '.' and root != '.\.idea' and root != '.\.idea\scopes':
		print root
		print set(fil[::2] + fil[::3] + fil[::5])
		for f in set(fil[::2] + fil[::3] + fil[::5]):
			tempFile = list(open(os.path.join(root, f), 'r').read().split())[4:]
			#print tempFile
			print >> out, label,
			for i in range(1, len(tempFile) + 1):
				print >> out, str(i) + ':' + tempFile[i-1],
			print >> out
		label += 1
label = 0
out.close()
out = open('test_data', 'w')

print 'constructing testing data...'

# construct testing data
for root, dirs, fil in os.walk('.'):
	if root != '.' and root != '.\.idea' and root != '.\.idea\scopes':
		print root
		print set(fil[::]) - set(fil[::2] + fil[::3] + fil[::5])
		for f in set(fil[::]) - set(fil[::2] + fil[::3] + fil[::5]):
			tempFile = list(open(os.path.join(root, f), 'r').read().split())[4:]
			#print tempFile
			print >> out, label,
			for i in range(1, len(tempFile) + 1):
				print >> out, str(i) + ':' + tempFile[i-1],
			print >> out
		label += 1
label = 0

out.close()

y, x = svm_read_problem('train_data')
print 'training instances...'
model = svm_train(y, x, '-q -t 0')
a, b = svm_read_problem('test_data')

print 'testing data...'
classification, acc, val = svm_predict(a, b, model)
print classification
print acc
print val
