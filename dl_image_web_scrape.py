import os
import json
import requests # to sent GET requests
from bs4 import BeautifulSoup # to parse HTML

# user can input a topic and a number
# download first n images from google image search

GOOGLE_IMAGE = \
    'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'

# The User-Agent request header contains a characteristic string
# that allows the network protocol peers to identify the application type,
# operating system, and software version of the requesting software user agent.
# needed for google search
usr_agent = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
}

SAVE_FOLDER = 'fire_hydrants'


def main():
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    download_images()


def download_images():
    # ask for use input
    data = input('What are you looking for? ')
    n_images = int(input('How many images do you want to download? '))

    print('Start searching.....')

    # get url query string
    searchurl = GOOGLE_IMAGE + 'q=' + data
    print(searchurl)

    # request url, without usr_agent, the permission gets denied
    response = requests.get(searchurl, headers=usr_agent)

    # find all divs where class='rg_i'
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('img', {'class': 'rg_i Q4LuWd'}, limit=n_images)

    # extract the link from the tag
    links = [res.attrs.get('data-src', "No image url") for res in results]

    print(f'Downloading {len(links)} images....')

    # Access the data URI and download the image to a file
    for i, link in enumerate(links):
        if link == "No image url":
            continue
        response = requests.get(link)

        image_name = SAVE_FOLDER + '/' + data + str(i + 1) + '.jpg'
        with open(image_name, 'wb') as fh:
            fh.write(response.content)

    print('Done')


if __name__ == '__main__':
    main()
