#!/bin/bash


dir=$1

for file in "$dir/"*
  do
    if [[ -x $file ]]
    then
      idat64 -B -S"ida.py" $file
    else
      echo "$file"
    fi
done

