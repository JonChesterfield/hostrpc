#!/bin/bash

rm -rf acm && mkdir acm && cd acm

mkdir icppworkshops21-13
cd icpp*

mkdir pdf
cp ../../llpp_2021_shmem.pdf pdf/

mkdir Source
cp ../../main.tex Source/
cp ../../reference.bib Source/

mkdir supplements

cd -

zip -r icppworkshops21-13.zip icppworkshops21-13
