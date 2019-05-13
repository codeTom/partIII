#!/bin/bash

cd ../ml/classifier/
positives="0 10 54 56 86"
negatives="98 376 559"
rows="model1/reconstruction_test_s/ model1/combined_smooth_rw1  model1/combined_smooth_rw1100 model1/combined_smooth_rw2900 deeper7_fix3"
# model1/combined_smooth_scratch"

cols=""
for p in $positives
do
    cols="$cols positive/$p"
done
for n in $negatives
do
    cols="$cols negative/$n"
done
#first row has originals
fr=""
for c in $cols
do
    fr="$fr model1/reconstruction_test_s/${c}_o.png"
done
#echo $fr
convert +append $fr /tmp/row0.png
i=1
for row in $rows
do
    rowfiles=''
    for c in $cols
    do
        rowfiles="$rowfiles $row/${c}_r.png"
    done
    echo $rowfiles
    convert +append $rowfiles /tmp/row$i.png
    i=$(( $i + 1 ))
done
#only works for <10 rows
cd -
rm -f graphics/reconstructions_model1.png
convert -append /tmp/row*.png graphics/reconstructions_model1.png
rm -f /tmp/row*.png
#echo $cols
