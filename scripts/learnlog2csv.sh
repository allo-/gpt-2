#!/usr/bin/env bash

cat $1| grep "^\[" |sed 's/\[//;s/| .*loss=//;s/ avg=/ /'

# gnuplot command line:
# f(x)=a*x+b; fit f(x) "loss.txt" using 0:2 via a,b; plot "loss.txt" using 1:2 t "loss", "loss.txt" using 1:3 t "averaged loss", f(x)
#
# An example for skipping the first few data points and calculating the regression using only only the last data points:
# f(x)=a*x+b; fit [5000:7000] f(x) "loss.txt" using 0:2 via a,b; plot [500:] "loss.txt" using 1:2 t "loss", "loss.txt" using 1:3 t "averaged loss", f(x)
