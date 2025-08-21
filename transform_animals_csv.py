#!/usr/bin/env python3
import csv
from collections import defaultdict

def transform_animals_csv():
    animals_by_category = defaultdict(list)
    
    with open('animals_snafu_scheme.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        
        for row in reader:
            if len(row) >= 2:
                category = row[0]
                animal = row[1]
                animals_by_category[category].append(animal)
    
    with open('animals_by_category.txt', 'w', encoding='utf-8') as output_file:
        for category in sorted(animals_by_category.keys()):
            animals = sorted(animals_by_category[category])
            animals_str = ', '.join(animals)
            output_file.write(f"{category}: {animals_str}\n")
    
    print(f"Successfully transformed CSV to animals_by_category.txt")
    print(f"Found {len(animals_by_category)} categories with {sum(len(animals) for animals in animals_by_category.values())} total animals")

if __name__ == "__main__":
    transform_animals_csv()