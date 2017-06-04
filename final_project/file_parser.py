#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Quora Question Data Parser                   #
# Nathan Brahmstadt and Jordan Crane           #
################################################

class QuestionPair:
    def __init__(self, string):
    
        string = string.split('","')
        self.id = string[0].replace("\"","")
        self.qid1 = string[1]
        self.qid2 = string[2]
        self.q1 = regularize_string(string[3])
        self.q2 = regularize_string(string[4])
        self.is_duplicate = string[5].replace("\"","")
    
        
def regularize_string(string):
    string = string.lower()
    slist = list(string)
    punctuation = ['?', '/', '-', '.', '[', ']', '(', ')', '"', ',']
    for i,c in enumerate(slist):
        if c in punctuation:
            slist[i] = ' '
    return "".join(slist) 
    
def main():
    train_file = open('data/train.csv')
    output_file = open('data/sentences.csv','w')
    #Skip first line
    train_file.readline()
    
    data = []
   
    while True:
        #Some pairs are spread over 2 lines
        
        line = train_file.readline()
      
        if not line[-2:] == '\"\n':
            print line
            line_part2 = train_file.readline()
            print line_part2
            line = line[:-2] + line_part2
            print line
        if not line:
            break
        
        data.append(QuestionPair(line))
        output_file.write(data[-1].q1 + "," + data[-1].q2 + "," + data[-1].is_duplicate)
        
    
main()   