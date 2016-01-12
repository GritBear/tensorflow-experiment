import io_util

trainFile = 'train.csv'
testFile = 'test.csv'

TrainInput, header = io_util.extract_txt_arr(trainFile)

print header
print TrainInput

ID = TrainInput[:,0]
Target = TrainInput[:,-1]
Train = TrainInput[:,1:-1]

print ID
print Target
print Train

TestInput, header = io_util.extract_txt_arr(testFile)
