"""
0 abstract_painting
1 cityscape
2 genre_painting
3 illustration
4 landscape
5 nude_painting
6 portrait
7 religious_painting
8 sketch_and_study
9 still_life
"""
import os
import csv
import shutil  
from tqdm import tqdm


def main():
	count=0
	with open('../wikiart-csv/genre_train.csv', 'r') as csvfile:
		data = csv.reader(csvfile)
		for row in tqdm(data):
			if(row[1] == "6"):
				try:
					pathSource = "../wikiart/" + row[0]
					tmp = row[0].split('/')
					pathDest = "portrait/" + tmp[-1]
					shutil.copyfile(pathSource, pathDest)
				except:
					pass				


main()