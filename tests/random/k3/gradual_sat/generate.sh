#!/bin/bash

for f in `ls ../sat/*.cnf`; do
    ./gnovelty+ "$f" | grep 0$ > x.sol
    cat x.sol | grep -oE -- '-?[0-9]+' | while IFS= read -r num_str; do if (( num_str != 0 )); then echo "$num_str 0" ; fi ; done > x.units


    
    TOTAL_LINES=`wc -l x.units | awk '{print $1}'`
    CURRENT_LINES_TO_PRINT=0
    INCREMENT=10
    
    while true; do
	CURRENT_LINES_TO_PRINT=$((CURRENT_LINES_TO_PRINT + $INCREMENT))

	TARGET=`echo "$f" | awk -F"/" '{print $NF}' | sed s/"\.cnf"/"\.sl${CURRENT_LINES_TO_PRINT}\.cnf"/g`
	CLAUSES=`cat "$f" | grep ^p | awk '{print $4}'`
	XCLAUSES=$((CLAUSES + $CURRENT_LINES_TO_PRINT))
	cat "$f" | grep ^p | sed s/"${CLAUSES}"/"${XCLAUSES}"/g > $TARGET
	cat "$f" | grep -v ^p | grep -v ^c >> $TARGET
	head -n "$CURRENT_LINES_TO_PRINT" x.units >> $TARGET
	if [[ "$CURRENT_LINES_TO_PRINT" -ge "$TOTAL_LINES" ]]; then
	     break;
        fi
    done
done

			    
