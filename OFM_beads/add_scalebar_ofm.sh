#!/bin/bash
#usage: ./add_scalebar_ofm.sh inputimage outputimage
# [length_um [position]]
#length=100
#if [ $3 ]
#then
#    length=$3
#fi
width=`identify $1  | cut -d' ' -f3 | cut -d'x' -f1`
height=`identify $1  | cut -d' ' -f3 | cut -d'x' -f2`

#select the right scale bar length
if [ $width -gt 1000 ]
then
    ll=$(( 239 * 2 ))
    spx=`echo "scale=0; $width/1.3" | bc`
    spy=`echo "scale=0; $height/1.3" | bc`
    spex=$(( $ll + $spx ))
    spey=$(( $spy + 50 ))
    pointsize=140
    textx=$(( $spx + 30 ))
    texty=$(( $spey + 140 ))
    lt="20 µm"
else
    ll=24
    spx=`echo "scale=0; $width/1.3" | bc`
    spy=`echo "scale=0; $height/1.3" | bc`
    spex=$(( $spx + $ll ))
    spey=$(( $spy + 7 ))
    pointsize=12
    textx=$(( $spx - 3 ))
    texty=$(( $spey + 12 ))
    lt="1 µm"
fi

echo "$1->$2"
convert $1 -fill black -draw "polygon $spx,$spy $spex,$spy $spex,$spey $spx,$spey" -pointsize $pointsize  -draw "text $textx,$texty '$lt'" $2
