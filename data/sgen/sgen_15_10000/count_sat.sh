#!/bin/bash
num_sat=0
num_unsat=0
for i in {0..9999}; do
	file="sgen-$i.out"
        while IFS= read -r line; do
                if [[ "$line" == s* ]]
                then
                        if [[ "$line" == *UNSATISFIABLE* ]]
                        then
				echo 0 > "sgen-$i.label"
                                num_unsat=$((num_unsat + 1))
                        elif [[ "$line" == *SATISFIABLE* ]]
                        then
				echo 1 > "sgen-$i.label"
                                num_sat=$((num_sat + 1))
                        fi
                fi
        done < "$file"
done
echo  $num_sat
echo  $num_unsat
