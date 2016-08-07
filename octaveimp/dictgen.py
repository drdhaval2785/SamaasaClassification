# -*- coding: utf-8 -*-
import sys, re
import codecs
import string
import datetime
"""
Usage - python dictgen.py step2.csv step3.csv dict.txt class.txt
Creates a dictionary and index of unique words in step2.csv. 
The replacements are stored in step3.csv.
dict.txt file has the dictionary.
class.txt file has unique classes.
"""

# Function to return timestamp
def timestamp():
	return datetime.datetime.now()

def readcsv(csvfile):
	output = []
	for line in open(csvfile):
		line = line.strip()
		word,micro,macro = line.split(',')
		output.append((word,micro,macro))
	return output
def writedict(readcsvdata,dictfile):
	output = []
	diction = codecs.open(dictfile,'w','utf-8')
	for (word,micro,macro) in readcsvdata:
		output += word.split('-')
	output = list(set(output))
	diction.write('\n'.join(output))
	diction.close()
def findindex(word,diction):
	lendict = xrange(len(diction))
	for i in lendict:
		line = diction[i].strip()
		if word == line:
			return i
	else:
		return 0
def repdict(readcsvdata,step3file,dictfile,classfile):
	step3 = codecs.open(step3file,'w','utf-8')
	diction = codecs.open(dictfile,'r','utf-8').readlines()
	log = codecs.open('log.txt','a','utf-8')
	log.write('==========More than two parts in compound==========\n')
	counter = 0
	classtypes = []
	for (word,micro,macro) in readcsvdata:
		classtypes.append(macro)
	classfout = codecs.open(classfile,'w','utf-8')
	classtypes = list(set(classtypes))
	classfout.write('\n'.join(classtypes))
	classfout.close()
	for (word,micro,macro) in readcsvdata:
		wordsplit = word.split('-')
		if len(wordsplit) == 2:
			counter += 1
			word1, word2 = word.split('-')
			ind1 = findindex(word1,diction)
			ind2 = findindex(word2,diction)
			classrep = classtypes.index(macro)
			step3.write(str(ind1)+','+str(ind2)+','+str(classrep)+'\n')
			if counter % 100 == 0:
				print counter
		else:
			log.write(word+','+micro+','+macro+'\n')
	log.close()
	step3.close()
if __name__=="__main__":
	fin = sys.argv[1]
	fout = sys.argv[2]
	dictfile = sys.argv[3]
	classfile = sys.argv[4]
	readcsvdata = readcsv(fin)
	print len(readcsvdata), "entries in step2.csv"
	writedict(readcsvdata,dictfile)
	repdict(readcsvdata,fout,dictfile,classfile)
	step3data = codecs.open(fout,'r','utf-8').readlines()
	print len(step3data), "entries in step3.csv"
	classtypes = codecs.open(classfile,'r','utf-8').readlines()
	print len(classtypes), "types of class in data"