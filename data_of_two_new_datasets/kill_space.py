# ecoding=utf-8

ifn = r"label_just_race.txt"
ofn = r"real_race_label_short.txt"

infile = open(ifn,'rb')
outfile = open(ofn,'wb')

for eachline in infile.readlines():
        
        lines = filter(lambda ch: ch not in ' ', eachline) 

        outfile.write(lines)

infile.close
outfile.close








