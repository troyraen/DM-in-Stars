
#!/bin/bash

# array to hold line count of history file
# index is mass*10 (e.g. the index for m4p3 is 43)
declare -a hist_wcl=( $(for i in {1..50}; do echo 0; done) )

for all directories m*p*:
        path="m*p*/LOGS/history.data"
        masstxt=
        if path exists:
                numlines filename <<< $(wcl path)
                write to file "mp* history file has "




note:
${string:position:length}
Extract $length of characters substring from $string starting from $position. In the below example, first echo statement returns $
from: https://www.thegeekstuff.com/2010/07/bash-string-manipulation
