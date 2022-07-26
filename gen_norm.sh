#!/bin/bash

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



