import os


directory = 'Data/'

# Starting Indexing number
no = 0
for filename in os.listdir(directory):
    if filename.startswith('mask'):
        os.rename(directory + filename, "M_{number}.jpg".format(number=no))
        no += 1
        print(filename)
    elif filename.startswith('rect'):
        os.remove(directory + filename)

print("{number_files} files found".format(number_files= no))
