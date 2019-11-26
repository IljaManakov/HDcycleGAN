import argparse
import os
import zipfile

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

    filename = 'oct_quality.zip'

    gdd.download_file_from_google_drive(file_id='19p1KDG2j93mBJp9O_yenwMHQMaqgJiWG',
                                        dest_path=os.path.join(FLAGS.data_dir, filename),
                                        unzip=False)

    print('Unpacking...')

    with zipfile.ZipFile(os.path.join(FLAGS.data_dir, filename), 'r') as zip_ref:
        zip_ref.extractall(FLAGS.data_dir)

    print('Cleaning up...')

    os.remove(os.path.join(FLAGS.data_dir, filename))

    print("Finished downloading files for HDCycleGan to {}".format(FLAGS.data_dir))


if __name__ == '__main__':
    main()
