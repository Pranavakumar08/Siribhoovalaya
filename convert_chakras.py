import pandas as pd

def extract_all_chakras():
    # Read the Excel file
    df = pd.read_excel("Adhyaya_One_Chakras.xls", sheet_name='Chapter-1', header=None)
    
    all_chakras = []
    current_chakra = []
    
    # Iterate through all rows and build chakras
    for idx, row in df.iterrows():
        # Skip header rows
        if idx < 2:
            continue
            
        # Extract data from columns C to K (indices 2 to 10)
        row_data = []
        for col in range(2, 11):
            val = row[col]
            if pd.notna(val) and str(val).strip() != '':
                try:
                    # Handle values like '4-52' by taking first part
                    if '-' in str(val):
                        row_data.append(int(str(val).split('-')[0]))
                    else:
                        row_data.append(int(val))
                except:
                    row_data.append(0)  # Default to 0 if conversion fails
        
        # Add to current chakra if we have data
        if row_data:
            current_chakra.append(row_data)
            
        # When we have 27 rows of data, save as a chakra
        if len(current_chakra) == 27:
            all_chakras.append(current_chakra)
            current_chakra = []
            
        # Stop when we have 27 chakras
        if len(all_chakras) == 27:
            break
    
    return all_chakras

# Extract all chakras
chakras = extract_all_chakras()

# Print each chakra separately
for i in range(27):
    print(f"chakra1_{i+1} = {chakras[i]}")
    print()