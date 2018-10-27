#!/bin/bash                                                                     
INPUT_DIR=../bin/50x50/density/
OUTPUT_DIR=../bin/50x50/density_out/

mkdir $OUTPUT_DIR
SUFFIX=.png

FILES=`ls  $INPUT_DIR | sort -n`
count=0
for file in $FILES
do
#    if [ $file != "." ] && [ $file != ".." ]; then
        in_file=$INPUT_DIR$file
	filenum=`printf %03d $count`
	filename="image_${filenum}"
        out_file=$OUTPUT_DIR$filename$SUFFIX
        echo "$in_file"
        echo "$out_file"
        python mkplot.py $in_file $out_file
	count=`expr $count + 1`
#    fi
done

