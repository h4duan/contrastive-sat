#!/bin/bash

for i in {0..9999}
do
	echo $i
	input="sgen-${i}.cnf"
	output="sgen-${i}.out"
	./cryptominisat5_amd64_linux_static $input > $output
done
