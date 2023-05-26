import os

script_path = os.path.abspath(__file__)

# Get the directory of the script
script_directory = os.path.dirname(script_path)

print(script_directory)

# Specify the filename
filename = "model.pth"

# Combine the script directory and filename to get the full file path
file_path = os.path.join(script_directory, f'model_{200}.pth')
print(file_path)
