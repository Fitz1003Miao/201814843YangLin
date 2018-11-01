import os
import numpy as np
import shutil

raw_path = "../20news-18828"
train_path = "../data/train"
cv_path = "../data/cv"
test_path = "../data/test"

dirs = os.listdir(raw_path)
for dir in dirs:
    files = os.listdir(os.path.join(raw_path, dir))
    indexs = np.random.permutation(len(files))

    if not os.path.exists(os.path.join(train_path, dir)):
        os.mkdir(os.path.join(train_path, dir))
    if not os.path.exists(os.path.join(cv_path, dir)):
        os.mkdir(os.path.join(cv_path, dir))
    if not os.path.exists(os.path.join(test_path, dir)):
        os.mkdir(os.path.join(test_path, dir))

    for i in range(len(indexs)):
        old_file = os.path.join(os.path.join(raw_path, dir), files[indexs[i]])
        if i / len(indexs) <= 0.6:
            new_file = os.path.join(os.path.join(train_path, dir), files[indexs[i]])
            shutil.copyfile(old_file, new_file)

        elif i / len(indexs) <= 0.8:
            new_file = os.path.join(os.path.join(cv_path, dir), files[indexs[i]])
            shutil.copyfile(old_file, new_file)

        else:
            new_file = os.path.join(os.path.join(test_path, dir), files[indexs[i]])
            shutil.copyfile(old_file, new_file)
