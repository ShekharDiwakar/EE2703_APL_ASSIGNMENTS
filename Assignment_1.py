"""
Applied Program Lab 
Assignment 1
Name : Shekhar Diwakar
Roll no : EE20B123
INPUT: .netlist file
OUTPUT: Identifies errors in SPICE program code, and displays the tokens in reverse order
"""

from sys import argv, exit



CIRCUIT = ".circuit"
END = ".end"


def line2tokens(spiceLine):
    allWords = spiceLine.split()

    
    #R, L and C are  Independent Sources.
    
    if len(allWords) == 4:
        elementName = allWords[0]
        node1 = allWords[1]
        node2 = allWords[2]
        value = allWords[3]
        return [elementName, node1, node2, value]

    # CCxS
    elif len(allWords) == 5:
        elementName = allWords[0]
        node1 = allWords[1]
        node2 = allWords[2]
        voltageSource = allWords[3]
        value = allWords[4]
        return [elementName, node1, node2, voltageSource, value]

    # VCxS
    elif len(allWords) == 6:
        elementName = allWords[0]
        node1 = allWords[1]
        node2 = allWords[2]
        voltageSourceNode1 = allWords[3]
        voltageSourceNode2 = allWords[4]
        value = allWords[5]
        return [elementName, node1, node2, voltageSourceNode1, voltageSourceNode2, value]

    else:
        return []


def getTokens(lines):
    lines_token = []
    for i in range(0, len(lines)):  # iterate over valid range
        line = (
            lines[i].split("#")[0].split()
        )  # Removing the comment and spliting the line into words

        lines_token.append(
            line
        )  # Joining words after reversing and adding "\n" at the end of line

    return lines_token

def printOut(lines):
    output = ""
    for i in reversed(range(0, len(lines))):  # iterate over valid range
        line = (
            lines[i].split("#")[0].split()
        )  # removing comments and split line into words
        
        
        line.reverse() #reversing the list
        #this function takes care of the reversing the list
        
        
        
        output = output + (
            " ".join(line) + "\n"
        )  #join words after reversing and add "\n" at the end of line for new line

    print(output)


if len(argv) != 2:
    print("Invalid operation !")
    print(f"Usage: {argv[0]} <inputfile>'")
    exit()

try:
    with open(argv[1]) as f:
        lines = f.readlines()
        start = -1
        end = -2
        cnt_start = 0
        cnt_end = 0 
        
        
        
        for line in lines:  #extracting circuit definition i.e, start and end lines
            if CIRCUIT == line[0 : len(CIRCUIT)]:
                start = lines.index(line)
                cnt_start = cnt_start + 1
            elif END == line[: len(END)]:
                end = lines.index(line)
                cnt_end = cnt_end + 1
        
        
        
        if (cnt_start > 1 or cnt_end > 1):   #Checking if there are multiple .circuit/.end declarations in input file
            print(
                "Invalid circuit definition! Multiple .circuit/.end declarations detected"
            )
            exit(0)

        Lines = lines[start + 1 : end]
        LinesToken = getTokens(Lines)
        printOut(Lines)


except IOError:
    print("Invalid file!")
    exit()