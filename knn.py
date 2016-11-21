# encoding: utf-8

from numpy import *
import operator
import random

# 预处理
def preproces(rawDataPath, cleanDataPath):
    outstream = ""
    dataSet = []
    sample = []
    with open(rawDataPath, 'r') as f:
        f.readline()
        for line in f.readlines():
            line = line.split(',')
            sample = line[:6]
            sample.append(line[7])
            if "" not in sample:
                # sample[6] = TNMDict[sample[6]]
                dataSet.append(sample)

    for sample in dataSet:
        for x in sample:
            outstream += x + ','
        outstream = outstream.strip(',') + '\n'

    with open(cleanDataPath, 'w') as f:
        f.write(outstream)

# 交叉验证法，第i份化入测试集，其他九分化为训练集
def divDataSet(cleanDataPath, i):
    trainData = {"group":[],"labels":[]}
    testData = {"group":[],"labels":[]}
    group,labels = createDataSet(cleanDataPath)
    # 随机划分
    dataSetIndicies = list(range(len(group)))
    random.shuffle(dataSetIndicies)

    if i >= 0 and i < 10:
        trainDataIndex = dataSetIndicies[:len(group) // 10 * i]
        trainDataIndex.extend(dataSetIndicies[len(group) // 10 * (i+1):])
        testDataIndex = dataSetIndicies[len(group) // 10 * i:len(group) // 10 * (i+1)]

        for index in trainDataIndex:
            trainData["group"].append(group[index])
            trainData["labels"].append(labels[index])
        trainData["group"] = array(trainData["group"])

        for index in testDataIndex:
            testData["group"].append(group[index])
            testData["labels"].append(labels[index])
        testData["group"] = array(testData["group"])
    else:
        print("Num should be a int in 0..9")

    return trainData,testData

def createDataSet(dataPath):
    group = []
    labels = []
    with open(dataPath,'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            groupItem = list(map(float, line[:-1]))
            group.append(groupItem)
            labels.append(line[-1])
    dataSet = array(group)
    return dataSet, labels

# 计算两个向量的lp范数
def getLpDistances(lp, inX, dataSet):
    if not isinstance(lp, int):
        raise TypeError("lp must be a integer.")
    dataSetSize = dataSet.shape[0]

    diffMat = absolute(tile(inX, (dataSetSize, 1)) - dataSet)

    lpDiffMat = diffMat ** lp
    lpDistances = lpDiffMat.sum(axis = 1)
    distances = lpDistances ** (1.0/lp)
    return distances

def classify(inX, dataSet, labels, k, lp):
    Distances = getLpDistances(lp,inX,dataSet)
    sortedDistIndicies = Distances.argsort()
    classCount = {}

    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def dataClassTest(cleanDataPath, k, lp,i):
    trainData, testData = divDataSet(cleanDataPath, i)

    trainSet = trainData["group"]
    trainLabels = trainData["labels"]
    testSet = testData["group"]
    testLabels = testData["labels"]

    testSetSize = testSet.shape[0]
    errorCount = 0
    testCount = 0

    for inX in testSet:
        classifierResult = classify(inX, trainSet, trainLabels, k, lp)
        if classifierResult == testLabels[testCount]:
            pass
            # print("[Correct Classification] the classifier came back with : %s, the real answer is: %s"\
            #       % (classifierResult,testLabels[testCount]))
        else:
            errorCount += 1
            # print("[Wrong Classification] the classifier came back with : %s, the real answer is: %s" \
            #       % (classifierResult, testLabels[testCount]))
        testCount += 1
    # print("[testNum: %d, k: %d, lp: %d] The error rate is :%.3f%%" %(testNum,k,lp,errorCount/float(testSetSize)*100))
    return errorCount/float(testSetSize)

def crossValidation(cleanDataPath, k, lp):
    totalRate = 0.0
    for i in range(10):
        totalRate += dataClassTest(cleanDataPath, k, lp, i)
    totalRate /= 10
    return totalRate

def findBestArgs(cleanDataPath, maxK, maxLp):
    minRecord = {"k":None,"lp":None,"minErrorRate":1}   # 记录最小错误率以及对应的k,lp
    for k in range(1,maxK+1):
        for lp in range(1,maxLp+1):
            totalRate = crossValidation(cleanDataPath, k, lp)
            print("totalRate: %f%%, k: %d, lp: %d" %(totalRate*100,k,lp))
            if totalRate < minRecord["minErrorRate"]:
                minRecord['k'] = k;
                minRecord['lp'] = lp;
                minRecord['minErrorRate'] = totalRate;
    return minRecord


def hello():
    return  "hello"


if __name__ == "__main__":
    rawDataPath = "C:/Users/Mr.x/repos/DataMiningProject/zanwen/data/rawdata.csv"
    cleanDataPath = "C:/Users/Mr.x/repos/DataMiningProject/zanwen/data/cleandata.csv"

    testNum=100
    maxK= 5
    maxLp= 5
    minRecord = findBestArgs(cleanDataPath, maxK, maxLp)
    print("Minimal error rate: %f%%, when k: %d, lp: %d"%(minRecord['minErrorRate']*100,minRecord['k'], minRecord['lp']))

