# import argparse
# import os
# from zipfile import ZipFile
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--dest', '-d', default='./oct_quality.zip', dest='dest')
# args = parser.parse_args()
#
# # download zip
# id = '19p1KDG2j93mBJp9O_yenwMHQMaqgJiWG'
# os.system(f"gdown 'https://drive.google.com/uc?&id={id}' -O {args.dest}")
# print(f'dataset downloaded to {args.dest}')
#
# # extract zip archive
# print('extracting files...')Dockerfile
# requirements.txt
# with ZipFile(args.dest, 'r') as file:
#     file.extractall(path=os.path.dirname(args.dest))
#
# print('all done!')

import argparse
import os
import tarfile

from google_drive_downloader import GoogleDriveDownloader as gdd

PARSER = argparse.ArgumentParser(description="V-Net medical")

PARSER.add_argument('--data_dir',
                    type=str,
                    default='./data',
                    help="""Directory where to download the dataset""")


def main():
    FLAGS = PARSER.parse_args()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    gdd.download_file_from_google_drive(file_id='19p1KDG2j93mBJp9O_yenwMHQMaqgJiWG',
                                        dest_path=os.path.join(FLAGS.data_dir, filename),
                                        unzip=False)

    print('Unpacking...')

    tf = tarfile.open(os.path.join(FLAGS.data_dir, filename))
    tf.extractall(path=FLAGS.data_dir)

    print('Cleaning up...')

    os.remove(os.path.join(FLAGS.data_dir, filename))

    print("Finished downloading files for V-Net medical to {}".format(FLAGS.data_dir))


if __name__ == '__main__':
    main()
