#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Quora Question Data Parser                   #
# Nathan Brahmstadt and Jordan Crane           #
################################################
from string import ascii_lowercase, digits

class QuestionPair:
    def __init__(self, string):
        string = string.split('","')
        if len(string) != 6:
            print len(string)
            print string
            #The stupid questions have a lot of edge cases for parsing. This solves the ones with "," in them
            for i,s in enumerate(string):
                while 1:
                    if s[-1] == '\"' and string[i+1][0] == '\"':
                        string[i] = string[i] + string[i+1]
                        print "Merged:"
                        print string[i]
                        del string[i+1]
                    else:
                        break
            print len(string)
            print string
        self.id = string[0].replace("\"","")
        self.qid1 = string[1]
        self.qid2 = string[2]
        self.q1 = regularize_string(string[3])
        self.q2 = regularize_string(string[4])
        self.is_duplicate = string[5].replace("\"","")
    
        
def regularize_string(string):
    string = string.lower()
    slist = list(string)
    punctuation = ['?', '/', '-', '.', '[', ']', '(', ')', '\"', ',', '\'', ':']
    for i,c in enumerate(slist):
        if c in punctuation:
            slist[i] = ' '
            
    for i,c in enumerate(slist):   
        #Remove extra whitespace
        repeat_flag = True
        while repeat_flag:
            repeat_flag = False
            if not i+1 == len(slist):
                if slist[i] == ' ' and slist[i+1] == ' ':
                    del slist[i+1]
                    repeat_flag = True
                    
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
       
            line_part2 = train_file.readline()
           
            line = line[:-2] + line_part2
          
        if not line:
            break
        
        data.append(QuestionPair(line))
        output_file.write(data[-1].q1 + "," + data[-1].q2 + "," + data[-1].is_duplicate)
        
    
main()   