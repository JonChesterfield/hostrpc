#!/bin/bash

FILE=main.tex
rm -f $FILE

touch $FILE
cat header >> $FILE
cat paper.txt >> $FILE
cat footer >> $FILE

