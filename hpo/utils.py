import os


# def find_files(root, filename):
#     for directory, subdirs, files in os.walk(root):
#         if filename in files:
#             return os.path.join(root, directory, filename)


def find_file(root, filename):
    """
    Recursively search for a file with the given name starting from the
    specified root directory.
    """
    # Check the root directory for the file
    if filename in os.listdir(root):
        return os.path.join(root, filename)

    # Search the subdirectories
    for dirname in os.listdir(root):
        path = os.path.join(root, dirname)
        if os.path.isdir(path):
            result = find_file(path, filename)
            if result is not None:
                return result
