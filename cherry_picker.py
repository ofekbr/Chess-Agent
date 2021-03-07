from itertools import zip_longest

file_list = ["masters_times.txt", "masters_clock.txt", "masters_available_moves.txt", "masters_moves.txt", "masters_materials.txt", "masters_taken.txt", "masters_threats.txt"]
new_file_list = ["new_times.txt", "new_clock.txt", "new_available_moves.txt", "new_count_moves.txt", "new_materials.txt", "new_taken.txt", "new_threats.txt"]

files = [open(filename) for filename in file_list]
new_files = [open(filename, 'w') for filename in new_file_list]
counter = 1
for lines in zip_longest(*files, fillvalue=''):
    lines = list(lines)
    if 0 <= int(lines[0][:-1]) <= 100:
        for i, file in enumerate(new_files):
            file.write(lines[i])
        counter += 1

for file in files:
    file.close()
for file in new_files:
    file.close()
