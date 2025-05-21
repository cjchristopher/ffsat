#!/bin/bash


for i in $(seq 30 10 400)
do
    for f in *.sl$i.* ; do time ./tinisat "$f" ; done
done

