import bayes

listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print("myVocabList = %s" % myVocabList)

testVec = bayes.setOfWords2Vec(myVocabList, listOPosts[0])
print("testVec=%s" % testVec)
