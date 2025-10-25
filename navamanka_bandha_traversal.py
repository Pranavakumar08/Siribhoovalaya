import numpy as np

class SiribhoovalayaDecryptor:
    def __init__(self, numerical_grid):
        # Fix inconsistent row lengths
        fixed_grid = []
        for row in numerical_grid:
            if len(row) < 27:
                padded_row = row + [0] * (27 - len(row))
                fixed_grid.append(padded_row)
            else:
                fixed_grid.append(row[:27])
        
        self.grid = np.array(fixed_grid)
        self.rows, self.cols = self.grid.shape
        
        # DIRECT mapping from numbers to Roman transliteration (ONLY REAL MAPPING)
        self.number_to_roman = {
            1: "a",  2: "ā",  3: "āā",  4: "i",  5: "ī",  6: "īī",  7: "u",  8: "ū",
            9: "ūū", 10: "ṛ", 11: "ṝ", 12: "ṝṝ", 13: "ḷ", 14: "l̤", 15: "l̤l̤", 16: "e",
            17: "ee", 18: "eee", 19: "ai", 20: "aī", 21: "aīī", 22: "o", 23: "ō", 24: "ōō",
            25: "au", 26: "aū", 27: "aūū", 28: "k", 29: "kh", 30: "g", 31: "gh",
            32: "ṅ", 33: "c", 34: "ch", 35: "j", 36: "jh", 37: "ñ", 38: "ṭ", 39: "ṭh",
            40: "ḍ", 41: "ḍh", 42: "ṇ", 43: "t", 44: "th", 45: "d", 46: "dh", 47: "n",
            48: "p", 49: "ph", 50: "b", 51: "bh", 52: "m", 53: "y", 54: "r", 55: "l",
            56: "v", 57: "ś", 58: "ṣ", 59: "s", 60: "h", 61: "ṁ", 62: "ḥ", 63: "ḵ", 64: "phk"
        }
    
    def number_to_text(self, number):
        """Convert number directly to Roman transliteration"""
        if number == 0:
            return ''  # Skip padding zeros
        if number in self.number_to_roman:
            return self.number_to_roman[number]
        else:
            return f'[{number}]'  # Mark unknown numbers
    
    def decrypt_sequence(self, number_sequence):
        """Convert sequence of numbers to Roman text"""
        result = []
        for num in number_sequence:
            if num != 0:
                result.append(self.number_to_text(num))
        return ''.join(result)
    
    # NAVAMANKA BANDHA PATTERNS
    def navamanka_spiral_center(self):
        """Spiral pattern from center - Navamanka Bandha"""
        result_numbers = []
        visited = np.zeros((self.rows, self.cols), dtype=bool)
        
        row, col = 13, 13  # Center of 27x27 grid
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dir_idx = 0
        step_size = 1
        steps_taken = 0
        
        for _ in range(self.rows * self.cols):
            if 0 <= row < self.rows and 0 <= col < self.cols and not visited[row, col]:
                result_numbers.append(self.grid[row, col])
                visited[row, col] = True
            
            # Move to next position
            row += directions[dir_idx][0]
            col += directions[dir_idx][1]
            steps_taken += 1
            
            # Change direction
            if steps_taken >= step_size:
                steps_taken = 0
                dir_idx = (dir_idx + 1) % 4
                if dir_idx % 2 == 0:
                    step_size += 1
        
        return self.decrypt_sequence(result_numbers)
    
    def navamanka_diagonal_nine(self):
        """Diagonal patterns with 9-step intervals"""
        result_numbers = []
        # Multiple diagonals starting from positions related to 9
        for start in [0, 9, 18]:
            for i in range(min(self.rows, self.cols - start)):
                result_numbers.append(self.grid[i, start + i])
        return self.decrypt_sequence(result_numbers)
    
    def navamanka_chakra_pattern(self):
        """Circular pattern moving outward from center"""
        result_numbers = []
        center_row, center_col = 13, 13
        
        # Read in concentric squares (simpler than circles)
        for radius in range(14):
            # Top row of square
            for j in range(-radius, radius + 1):
                row, col = center_row - radius, center_col + j
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    result_numbers.append(self.grid[row, col])
            # Right column
            for i in range(-radius + 1, radius):
                row, col = center_row + i, center_col + radius
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    result_numbers.append(self.grid[row, col])
            # Bottom row
            for j in range(radius, -radius - 1, -1):
                row, col = center_row + radius, center_col + j
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    result_numbers.append(self.grid[row, col])
            # Left column
            for i in range(radius - 1, -radius, -1):
                row, col = center_row + i, center_col - radius
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    result_numbers.append(self.grid[row, col])
        
        return self.decrypt_sequence(result_numbers)
    
    def analyze_common_sounds(self):
        """Analyze which sounds appear most frequently"""
        from collections import Counter
        
        all_numbers = self.grid.flatten()
        number_count = Counter(all_numbers)
        
        print("\nMost common sounds in grid:")
        print("-" * 30)
        for num, count in number_count.most_common(15):
            if num != 0 and num in self.number_to_roman:
                roman = self.number_to_roman[num]
                print(f"  {num:2d} -> {roman:4s}: {count:3d} times")

# Your input grid
chakra1_1 = [
    [59, 23, 1, 16, 28, 28, 1, 1, 56, 59, 4, 1, 1, 47, 16, 34, 1, 7, 16, 1, 1, 7, 56, 1, 60],
    [53, 54, 47, 28, 1, 47, 45, 28, 7, 4, 59, 41, 4, 45, 1, 30, 47, 47, 45, 42, 53, 28, 51, 1, 52, 1, 1],
    [1, 22, 1, 30, 2, 1, 2, 55, 30, 1, 7, 45, 47, 52, 1, 4, 1, 47, 1, 1, 1, 1, 53, 1, 52, 59, 52],
    [59, 30, 2, 55, 55, 13, 16, 2, 53, 60, 1, 4, 16, 47, 48, 45, 16, 56, 56, 43, 45, 1, 56, 1, 4, 1, 13],
    [47, 45, 1, 1, 22, 30, 51, 1, 2, 56, 38, 30, 4, 1, 1, 56, 1, 1, 16, 1, 57, 7, 56, 56, 1, 22, 1],
    [54, 52, 52, 45, 1, 7, 55, 48, 1, 58, 52, 35, 28, 55, 1, 38, 45, 30, 55, 4, 47, 7, 45, 38, 45, 38, 1],
    [1, 1, 1, 28, 13, 56, 55, 51, 54, 1, 1, 1, 1, 42, 2, 4, 4, 1, 43, 16, 47, 7, 1, 13, 4, 51, 4],
    [28, 53, 47, 22, 8, 1, 53, 59, 38, 7, 43, 40, 1, 52, 59, 54, 30, 1, 45, 16, 1, 28, 23, 50, 7, 43, 43],
    [1, 2, 45, 51, 30, 1, 52, 58, 48, 59, 47, 54, 4, 4, 1, 47, 45, 47, 56, 28, 1, 45, 1, 13, 7, 7, 7],
    [55, 1, 53, 47, 56, 1, 1, 7, 1, 1, 2, 60, 48, 56, 1, 1, 16, 1, 1, 54, 1, 52, 17, 30, 54, 45, 45],
    [59, 56, 52, 1, 45, 1, 55, 28, 52, 28, 1, 2, 1, 52, 54, 4, 43, 60, 48, 28, 1, 16, 23, 8, 53, 7, 1],
    [2, 1, 53, 52, 43, 23, 2, 4, 16, 52, 44, 54, 1, 2, 42, 7, 1, 7, 47, 30, 28, 48, 47, 1, 54, 52, 16],
    [45, 54, 23, 4, 28, 45, 45, 30, 1, 59, 1, 56, 28, 2, 54, 53, 38, 2, 2, 1, 28, 55, 40, 60, 4, 50, 28],
    [2, 13, 47, 1, 1, 4, 17, 45, 1, 56, 1, 52, 56, 51, 1, 47, 55, 55, 45, 7, 2, 54, 1, 56, 7, 1, 1],
    [23, 4, 53, 54, 59, 48, 13, 56, 1, 47, 23, 1, 2, 55, 16, 1, 1, 47, 40, 54, 16, 52, 1, 47, 60, 43, 60],
    [45, 16, 43, 1, 7, 47, 1, 7, 1, 4, 54, 54, 1, 43, 28, 28, 7, 1, 2, 7, 52, 30, 1, 4, 47, 4, 13],
    [42, 1, 54, 13, 1, 28, 1, 45, 42, 5, 48, 56, 1, 1, 1, 52, 54, 7, 1, 1, 2, 56, 56, 2, 43, 1, 1],
    [56, 43, 22, 45, 56, 43, 2, 2, 56, 1, 8, 48, 59, 59, 7, 16, 53, 55, 53, 48, 1, 1, 46, 2, 30, 53, 1],
    [47, 45, 1, 2, 54, 56, 56, 2, 55, 51, 4, 16, 7, 13, 30, 16, 1, 1, 4, 52, 52, 4, 54, 47, 2, 38, 1],
    [1, 54, 60, 56, 54, 1, 60, 1, 1, 16, 40, 38, 17, 1, 47, 56, 33, 55, 1, 1, 59, 48, 1, 53, 7, 1, 1],
    [1, 52, 16, 1, 60, 1, 30, 53, 30, 7, 47, 13, 13, 22, 8, 13, 45, 59, 54, 1, 2, 42, 54, 47, 53, 52, 53],
    [16, 30, 1, 4, 52, 47, 56, 1, 28, 16, 1, 22, 59, 51, 1, 1, 7, 28, 53, 60, 7, 1, 16, 16, 1, 1, 58],
    [4, 53, 56, 1, 52, 2, 13, 52, 38, 30, 45, 7, 1, 30, 56, 16, 1, 1, 1, 30, 48, 56, 54, 54, 55, 28, 45],
    [1, 47, 47, 1, 28, 22, 1, 47, 1, 1, 45, 46, 1, 1, 47, 53, 55, 52, 1, 1, 7, 43, 2, 1, 1, 1, 43],
    [1, 4, 53, 1, 45, 43, 16, 55, 52, 4, 47, 55, 45, 22, 51, 56, 1, 38, 13, 30, 2, 28, 56, 13, 56, 28, 55],
    [4, 16, 46, 1, 1, 16, 1, 1, 1, 1, 1, 47, 59, 4, 8, 38, 58, 1, 1, 48, 1, 7, 22, 1, 1, 1, 60],
    [52, 4, 30, 56, 53, 52, 54, 1, 30, 52, 1, 16, 54, 7, 58, 1, 30, 54, 1, 56, 51, 53, 56, 57, 56, 4, 60]
]

# Execute
if __name__ == "__main__":
    print("SIRIBHOOVALAYA DECRYPTION - SIMPLIFIED")
    print("=" * 70)
    print("Direct Number → Roman Transliteration Mapping")
    print("=" * 70)
    
    decryptor = SiribhoovalayaDecryptor(chakra1_1)
    
    print(f"Grid: {decryptor.rows} x {decryptor.cols}")
    
    # Show some sample mappings
    print("\nSample mappings:")
    print("-" * 20)
    for num in [1, 4, 7, 16, 28, 47, 56, 59]:
        if num in decryptor.number_to_roman:
            print(f"  {num:2d} → {decryptor.number_to_roman[num]}")
    
    # Analyze
    decryptor.analyze_common_sounds()
    
    print("\n" + "=" * 70)
    print("NAVAMANKA BANDHA DECRYPTION:")
    print("=" * 70)
    
    patterns = [
        ("1. Navamanka Spiral (Center)", decryptor.navamanka_spiral_center),
        ("2. Navamanka Diagonal-9", decryptor.navamanka_diagonal_nine),
        ("3. Navamanka Chakra", decryptor.navamanka_chakra_pattern),
    ]
    
    for name, method in patterns:
        print(f"\n{name}:")
        print("-" * 40)
        result = method()
        # Show output
        display_text = result #+ "..." if len(result) > 250 else result
        print(display_text)
        print(f"Length: {len(result)} characters")
    
    print("\n" + "=" * 70)