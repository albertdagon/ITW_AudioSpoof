import csv

with open('./release_in_the_wild/release_in_the_wild/meta.csv', 'r') as csv_file, open('in_the_wild.protocol.txt', 'w') as txt_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        row[1] = row[1].replace(" ", "")
        txt_file.write(' '.join(row) + '\n')