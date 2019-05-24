import io
import csv
from PIL import Image  # https://pillow.readthedocs.io/en/4.3.x/
import requests  # http://docs.python-requests.org/en/master/
from tqdm import tqdm
import traceback

path = 'dpf/data/dif/'
begin_with = 1827
failed_dict = {}

# example image url: https://m.media-amazon.com/images/S/aplus-media/vc/6a9569ab-cb8e-46d9-8aea-a7022e58c74a.jpg
def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert 'Status code error: {}.'.format(r.status_code)

    with Image.open(io.BytesIO(r.content)) as im:
        im.save(image_file_path)



with open(f'{path}/data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for i, row in tqdm(enumerate(csv_reader)):
        if i<begin_with:
            continue

        try:
            download_image(row[1], image_file_path=f'{path}/images/{i}.jpg')
        except Exception as e:
            print(f'Failed, last processed: {i}')
            failed_dict[i] = row[1]
            print(traceback.format_exc())

            continue
        line_count = i
    print(f'Processed {line_count} lines.')
    print(f'failed dict: {failed_dict}')