import os
from PIL import Image
import tqdm

checkdir = os.path.join("data", "style")
files = os.listdir(checkdir)
format = [".jpg", ".jpeg"]

for(path, dirs, f) in os.walk(checkdir):
    for file in tqdm.tqdm(f):
        if file.endswith(tuple(format)):
            try:
                image = Image.open(path+"/"+file).load()
                # print(image)
            except Exception as e:
                print("An exception is raised:", e)
                print(file)