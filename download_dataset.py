import argparse
import os
from zipfile import ZipFile

parser = argparse.ArgumentParser()
parser.add_argument('--dest', '-d', default='./oct_quality.zip', dest='dest')
args = parser.parse_args()

# download zip
id = '19p1KDG2j93mBJp9O_yenwMHQMaqgJiWG'
os.system(f"gdown 'https://drive.google.com/uc?&id={id}' -O {args.dest}")
print(f'dataset downloaded to {args.dest}')

# extract zip archive
print('extracting files...')
with ZipFile(args.dest, 'r') as file:
    file.extractall(path=os.path.dirname(args.dest))

print('all done!')
