# Remove case endings with trimmer.py and store output in step1.csv
python trimmer.py samAsa_details.csv step1.csv
# Remove duplicate lines from step1.csv and write to step2.csv
awk '!x[$0]++' step1.csv > step2.csv
# Create a dictionary dict.txt having all unique words from step2.csv
python dictgen.py step2.csv step3.csv dict.txt class.txt
