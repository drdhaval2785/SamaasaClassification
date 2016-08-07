# -*- coding: utf-8 -*-
import sys, re
import codecs
import string
import datetime
"""
Usage - python trimmer.py inputfile outputfile
e.g.
python trimmer.py samAsa_details.csv samAsa_trimmed.csv
"""

# Function to return timestamp
def timestamp():
	return datetime.datetime.now()
def trimvibhakti(word,vibhaktilist):
	for vib in vibhaktilist:
		if word.endswith(vib):
			outword = re.sub(vib+"$","",word)
			break
	else:
		outword = word
	return outword

def readcsv(csvfile):
	output = []
	for line in open(csvfile):
		line = line.strip()
		word,micro,macro = line.split(',')
		output.append((word,micro,macro))
	return output
if __name__=="__main__":
	# From Kale's Higher Sanskrit Grammar page 35-108
	vibhaktilist = ["aH","O","AH","am","aM","An","ena","eRa","EH","Aya","AByAm","AByAM","eByaH","At","yoH","asya","AnAm","AnAM","ARAm","ARAM","e","ezu","Am","AM","ayA","ABiH","AyE","AByaH","AyAH","AyAm","Asu","Ani","ARi","a","A","oH","i","E","iH","I","ayaH","im","iM","In","iRA","inA","iByAm","iBiH","iByaH","aye","eH","IRAm","IRAM","InAm","InAM","izu","IH","yA","yE","yAH","yAm","yAM","uH","U","avaH","o","um","uM","Un","uRA","unA","uByAm","uByAM","uBiH","ave","uByaH","voH","UnAm","UnAM","URAm","URAM","uzu","UH","vA","vE","vAH","vAm","vAM","i","iRI","inI","IRi","Ini","ine","iRe","inaH","iRaH","inoH","iRoH","ini","iRi","u","unI","uRI","Uni","URi","une","uRe","unaH","uRaH","unoH","uRoH","uni","uRi","u","A","AyO","AyaH","e","yuH","yO","yaH","i","Izu","vO","Um","UM","UByAm","UByAM","UBiH","UByaH","Uzu","Fn","rA","fByAm","fByAM","fBiH","fByaH","roH","FRAm","FRAM","ari","fzu","fRI","FRi","f","fRA","fRe","fRaH","fRoH","A","Im","an","Ad","IM"]
	vibhaktilist = list(set(vibhaktilist))
	vibhaktilist = sorted(vibhaktilist,key=lambda x:len(x),reverse=True)
	fin = sys.argv[1]
	fout = sys.argv[2]
	data = readcsv(fin)
	counter1 = 0
	counter2 = 0
	matchfile = codecs.open(fout,'w','utf-8')
	logfile = codecs.open('log.txt','w','utf-8')
	logfile.write("==========Unmatched words==========\n")
	for (word,micro,macro) in data:
		trimmedword = trimvibhakti(word,vibhaktilist)
		if trimmedword == word:
			counter2 += 1
			logfile.write(word+','+micro+','+macro+'\n')
			matchfile.write(word+','+micro+','+macro+'\n')
		else:
			counter1 += 1
			#matchfile.write(word+','+micro+','+macro+'\n;'+trimmedword+','+micro+','+macro+'\n')
			matchfile.write(trimmedword+','+micro+','+macro+'\n')
	matchfile.close()
	print counter1, "entries normalized."
	print counter2, "entries not normalized. Stored in log.txt in case you want to examine."
	print counter1+counter2, "total entries written to step1.csv"
	