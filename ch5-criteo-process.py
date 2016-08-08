import numpy
import random
import sys

def process_file (file_name, vw_file_name_train, vw_file_name_test, train_pct):
    file = open(file_name)
    vw_file_train = open(vw_file_name_train,'w')
    vw_file_test = open(vw_file_name_test,'w')
    
    continuous_set = [int(x) for x in numpy.linspace(1,13,13)]
    categorical_set =[int(x) for x in numpy.linspace(1,26,26)]
    
    print continuous_set
    print categorical_set
    
    first_line_headers = ["Class"]
    for i in continuous_set:
        first_line_headers.append("i"+str(i))
    
    for c in categorical_set:
        first_line_headers.append("c"+str(c))

    print first_line_headers

    for line in file:
        line_split = line.split('\t')
        target_click = -1
        if int(line_split[0])>0:
            target_click=1

        #Essentially now manually build up the training string
        vw_line = ""+str(target_click)+" "  

        for feature_index in continuous_set:
            if line_split[feature_index]!="":
                vw_line+="|"+first_line_headers[feature_index] +" c:"+ line_split[feature_index] + " "

        for feature_index in [x+len(continuous_set) for x in categorical_set]: #Index doesn't start from 0
            if line_split[feature_index]!="":
                vw_line+="|"+first_line_headers[feature_index] + " " + line_split[feature_index] + " "

        if(random.random()<=train_pct):
            vw_file_train.write(vw_line.replace('\n', '')+"\n") #Get rid of any unwanted line breaks
        else:
            vw_file_test.write(vw_line.replace('\n', '')+"\n") #Get rid of any unwanted line breaks

    file.close()
    vw_file_train.close()
    vw_file_test.close()

if __name__ == '__main__':
	filename='./train.txt'
	vw_file_name_train = './train_vw_file'
	vw_file_name_test = './test_vw_file'

	filename = sys.argv[1] if len(sys.argv) >=2 else filename
	vw_file_name_train = sys.argv[2] if len(sys.argv) >=3 else vw_file_name_train
	vw_file_name_test = sys.argv[3] if len(sys.argv) >=4 else vw_file_name_test
	
	process_file(filename,vw_file_name_train,vw_file_name_test,0.7)





