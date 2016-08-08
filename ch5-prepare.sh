#!/bin/bash

#This file contains the code for listing 5.1 - 5.5.
#This code can be run as is, if it placed in the same directory as the 
#train_vw_file and the location of your vowpal_wabbit checkout is correct.
#In order to obtain the train_vw_file you must run ch5-criteo-process.py 
#over the train.txt file obtained from the criteo display challenge dataset (full).

#!5.1
#echo 'Running Listing 5.1'
wc -l train_vw_file
grep -c '^-1' train_vw_file
grep -c '^1' train_vw_file

#!5.2
#echo 'Running Listing 5.2'
grep '^-1' train_vw_file | sort -R  > negative_examples.dat
grep '^1' train_vw_file | sort -R > positive_examples.dat	
awk 'NR % 3 == 0' negative_examples.dat > negative_examples_downsampled.dat

cat negative_examples_downsampled.dat > all_examples.dat
cat positive_examples.dat >> all_examples.dat

cat all_examples.dat | sort -R  > all_examples_shuffled.dat
awk 'NR % 10 == 0' all_examples_shuffled.dat > all_examples_shuffled_down.dat

#!5.3
echo 'Running Listing 5.3'
vw all_examples_shuffled_down.dat --loss_function=logistic -c -b 22 --passes=3 -f model.vw
vw all_examples_shuffled_down.dat -t -i model.vw --invert_hash readable.model
cat readable.model | awk 'NR > 9 {print}' | sort -r -g -k 3 -t : | head -1000 > readable_model_sorted_top

#!5.4
#Output only

#!5.5
echo 'Running Listing 5.5'
vw -d test_vw_file -t -i model.vw --loss_function=logistic -r predictions.out
~/dev/vowpal_wabbit/utl/logistic -0 predictions.out > probabilities.out
cut -d ' ' -f 1 test_vw_file | sed -e 's/^-1/0/' > ground_truth.dat


