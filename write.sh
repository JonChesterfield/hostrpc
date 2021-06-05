#!/bin/bash

FILE=main.tex
rm -f $FILE

MINIMAL=hostrpc/minimal

touch $FILE
cat header >> $FILE
cat paper.txt >> $FILE
cat footer >> $FILE

for i in header.cpp client.cpp server.cpp main.cpp; do
    sed -e "/$i/{r $MINIMAL/$i" -e "d}" $FILE > tmp
    mv tmp $FILE
done
