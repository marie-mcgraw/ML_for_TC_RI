import os.path

directories = ['out',
               'Figures',
               'REALTIME']
#
def setup_directory(target):
    if not os.path.isdir(target):
        os.mkdir(target)
        print(f"directory {target} has been created.")
    else:
        print(f"directory {target} already exists.")


def setup_directories():
    for target in directories:
        setup_directory(target)


if __name__ == "__main__":
    setup_directories()