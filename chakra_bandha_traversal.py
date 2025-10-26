import numpy as np
import csv
from roman import siribhoovalaya_roman
from chakra1_1 import chakra1_1

# Converting a nested list of numbers into a numpy array of corresponding alphabets.
def convert_grid_to_alphabetic(grid_numbers, mapping_dict):

    grid_alpha = [
        [mapping_dict.get(num, "?") for num in row]
        for row in grid_numbers
    ]
    return grid_alpha


def save_grid(grid_alpha, txt_filename="siribhoovalaya.txt", csv_filename="siribhoovalaya.csv"):
   
    # Save as TXT
    with open(txt_filename, "w", encoding="utf-8") as f:
        for row in grid_alpha:
            f.write(" ".join(row) + "\n")

    # Save as CSV
    with open(csv_filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(grid_alpha)

    print(f"Saved grid as:\n  TXT = {txt_filename}\n  CSV = {csv_filename}")


def traverse_grid(grid, start_row=0, start_col=13):
    n = len(grid)
    result = []
    visited = []
    r, c = start_row, start_col

    while True:
        result.append(grid[r][c])
        visited.append((r,c))

        if r == n-1 and c == 13:
            break

        # Move up-right
        r -= 1
        c += 1

        if (r,c) in visited:
            r += 1

        # Wrap around vertically if we move above the top
        if r < 0:
            r = n - 1

        # Stop if we move beyond the rightmost column
        if c >= n:
            c = 0

    return ''.join(result)


grid_numbers = chakra1_1


# Convert numeric to IAST Roman script
grid_alpha = convert_grid_to_alphabetic(grid_numbers, siribhoovalaya_roman) 

# Saving traversed grids
save_grid(grid_alpha, "siribhoovalaya_kannada.txt", "siribhoovalaya_kannada.csv")

output_text = traverse_grid(grid_alpha)
print("Output Text:", output_text)
