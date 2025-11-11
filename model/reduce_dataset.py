import os, random, shutil

source = r"C:\Users\bssad\Pictures\Projects\Multimedia\images\dataset\train_cleaned\merged"
target = r"C:\Users\bssad\Pictures\Projects\Multimedia\images\dataset\train_reduced"

os.makedirs(target, exist_ok=True)

for cls in os.listdir(source):
    cls_path = os.path.join(source, cls)
    imgs = os.listdir(cls_path)
    random.shuffle(imgs)
    keep = imgs[:2000]  # keep only 2000 per class (instead of 7215)
    
    dest_cls = os.path.join(target, cls)
    os.makedirs(dest_cls, exist_ok=True)
    for img in keep:
        shutil.copy(os.path.join(cls_path, img), os.path.join(dest_cls, img))

print(" Reduced dataset created.")
