#!/bin/bash
set -x
set -e
set -o pipefail

FILE=main.tex
rm -f $FILE

MINIMAL=minimal

touch $FILE
cat header >> $FILE
cat paper.txt >> $FILE
cat footer >> $FILE

rm -f a.out
clang++ -std=c++14 -Wall minimal.cpp -pthread -o a.out
./a.out

for i in header.cpp client.cpp server.cpp main.cpp; do
    sed -e "/$i/{r $MINIMAL/$i" -e "d}" $FILE > tmp
    mv tmp $FILE
done
