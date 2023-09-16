import os

def rename_folders(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if 'physical' in dirpath and 'atatck' in dirname:
                new_dirname = dirname.replace('atatck', 'attack')
                old_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, new_dirname)
                print("old_path: ", old_path)
                print("new_path: ", new_path)
                os.rename(old_path, new_path)

if __name__ == "__main__":
    root_directory = "/tmp/pycharm_project_250/larger_images"
    rename_folders(root_directory)
