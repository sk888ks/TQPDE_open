# Code to extract mean and variance from the output text of a QPDE (Quantum Phase Estimation) process
# After generating output like "python QPDE_Bayes_Hubbard.py > out.txt", execute "python get_prob_RD.py out.txt"

import re
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract data from a QPE output file.')
parser.add_argument('filename', type=str, help='Name of the file to process')
args = parser.parse_args()

# Initialization
prob_list = []
mu_list = []
var_list = []

# Open the file
with open(args.filename, 'r') as file:
    lines = file.readlines()

# Extract probability values
i = 0
while i < len(lines):
    if lines[i].startswith("eps val"):
        prob_list_temp = []
        for line in lines[i + 1: i + 22]:
            try:
                _, prob = line.split()
                prob_list_temp.append(float(prob))
            except ValueError:
                continue
        prob_list.append(prob_list_temp)
        i += 22  # Search for the next eps val
    else:
        i += 1

# Extract mean and variance
mean_variance_pattern = re.compile(r'Mean = (-?\d+\.\d+) Variance = (-?\d+\.\d+)')
for line in lines:
    if line.startswith("Iter"):
        match = mean_variance_pattern.search(line)
        if match:
            mu_list.append(float(match.group(1)))
            var_list.append(float(match.group(2)))

print(f"prob_list={prob_list}")
print(f"mu_list={mu_list}")
print(f"var_list={var_list}")
