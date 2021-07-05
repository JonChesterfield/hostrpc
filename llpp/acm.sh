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

# Maybe it needs this stuff in the source directory too?
# Doesn't have a license on it so assuming I can't paste it into the repo
wget https://www.acm.org/binaries/content/assets/publications/consolidated-tex-template/acmart-primary.zip

# Delete stuff to try to get back below their 10mb limit
unzip acmart-primary.zip -d .
rm -rf acmart-primary.zip acmart-primary/samples acmart-primary/.gitignore
mv acmart-primary/* Source/
rmdir acmart-primary

rm -rf acmart-primary.zip Source/acmart-primary/samples

cd -

zip -r $NAME.zip $NAME

du -sm $NAME.zip

