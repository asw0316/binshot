#!/bin/bash

################################################################
# Practical Binary Code Similarity Detection                   #
#   with BERT-based Transferable Similarity Learning           #
#   (In the 38th Annual Computer Security                      #
#    Applications Conference (ACSAC)                           #
#                                                              #
#  Author: Sunwoo Ahn <swahn@sor.snu.ac.kr>                    #
#          Dept. of Electrical and Computer Engineering        #
#            @ Seoul National University                       #
#          Hyungjoon Koo <kevin.koo@skku.edu>                  #
#          Dept. of Computer Science and Engineering           #
#            @ Sungkyunkwan University                         #
#                                                              #
#  This file can be distributed under the MIT License.         #
#  See the LICENSE file for details.                           #
################################################################

dir=$1
pkl=$2

for file in "$dir/"* 
    
  do
    if [[ -x $file ]]
    then
      python normalize.py $file $pkl
    else
      echo "$file"
    fi
done

