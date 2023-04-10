
import csv
from math import sqrt, pow, exp

def text_to_csv(input_file_name):
    with open(f"./YelpZip/{input_file_name}.txt", encoding='cp850') as source_file, \
            open(f"{input_file_name}.csv", mode='w', newline='', encoding='cp850') as target_file:
        
        csv_writer = csv.writer(target_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        for row in source_file:
            csv_writer.writerow(row.strip().split('\t'))

# Example usage
# transform_txt_to_csv('example_filename')
