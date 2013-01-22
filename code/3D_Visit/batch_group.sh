#!/bin/bash

visitFile="play.visit"
suffix=".*"

for (( prefix=0; prefix<=15; prefix++)) 
do
    for file in `ls $prefix$suffix | sort`
    do
        
        if [ $file != "." ] && [ $file != ".." ]; then
            echo $file >> $visitFile
        fi
    done
done
