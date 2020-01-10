#!/bin/sh

if [ -z "$2" ]
then
  echo "Usage: $0 <input dir> <output file>"
  echo "Concatenates txt files in <input dir> to plain utf-8 encoded text, adds newline markers and saves the file as <output file>."
  exit 1
fi

INPUT=$(readlink -f "$1")
OUTPUT=$(readlink -f "$2")
rm "$OUTPUT"
for f in "$INPUT"/*.txt
do
  cat "$f" | sed 's/^\s*//g;s/\s*$//g;s@<[a-zA-Z0-9/]*>@@g' | sed 's#$#<|n|>#g' >> "$OUTPUT"
  echo "" >> "$OUTPUT" # Insert a newline before endoftext marker
  echo "<|endoftext|>" >> "$OUTPUT"
done
