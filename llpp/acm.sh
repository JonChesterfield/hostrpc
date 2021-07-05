#!/bin/bash

rm -rf acm && mkdir acm && cd acm

NAME=icppworkshops21-13

mkdir $NAME
cd icpp*

mkdir pdf
cp ../../llpp_2021_shmem.pdf pdf/$NAME.pdf

mkdir Source
cp ../../main.tex Source/$NAME.tex
cp ../../reference.bib Source/

mkdir supplements

cd -

zip -r $NAME.zip $NAME
