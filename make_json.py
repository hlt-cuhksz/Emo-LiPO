
import json

# Function to process each line and convert it into a dictionary
def process_line(line):
    key, value = line.split(' ', 1)
    return key, value.strip()

# Initialize an empty dictionary to store the processed data
data = {}

# Read the content of a.txt and process each line
with open('text_esd', 'r') as file:
    for line in file:
        key, value = process_line(line)
        data[key] = [value]

# Write the processed data into b.txt in JSON format
with open('text_esd.json', 'w') as file:
    json.dump(data, file, indent=2)

