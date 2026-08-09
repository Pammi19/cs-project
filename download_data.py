import os
import urllib.request

# Ensure datasets directory exists
os.makedirs("datasets", exist_ok=True)

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

print("Downloading KDDTrain+.txt...")
urllib.request.urlretrieve(train_url, "datasets/KDDTrain+.txt")

print("Downloading KDDTest+.txt...")
urllib.request.urlretrieve(test_url, "datasets/KDDTest+.txt")

print("✅ NSL-KDD Dataset successfully downloaded to datasets/ folder!")