#!/bin/bash
for i in {0..100}; do
	file="dataset_distort/sgen-95-${i+100}.cnf"
	new_file="sgen_95_val/sgen-95-${i}.cnf"
	label="dataset_distort/sgen-95-${i+100}.label"
	new_label="sgen_95_val/sgen-95-${i}.label"
	cp "$file" "$new_file"
	cp "$label" "$new_label"
done
