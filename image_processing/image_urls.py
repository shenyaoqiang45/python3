import requests
# path = "/home/yaoqiang/Downloads/nsfw_data_scrapper/raw_data/sexy/123.jpg"
# url = 'http://cdn-s3.si.com/images/hailey5.jpg'
# r = requests.request('get',url)
# print(r.status_code)
# with open(path,'wb') as f:
#     f.write(r.content)
# f.close()


def getImagesTxt2(test_set, dstPath):
    f = open(test_set, 'r')
    i = 0
    lines = f.readlines()
    for url in lines:
        i = i + 1
        url = url.strip()
        r = requests.request('get', url)
        print(r.status_code)
        path = dstPath + str(i) + '.jpg'
        with open(path, 'wb') as f:
            f.write(r.content)
        f.close()
    return


if __name__ == "__main__":
    getImagesTxt2('/home/yaoqiang/Downloads/nsfw_data_scrapper/raw_data/sexy/urls_sexy.txt', '/home/yaoqiang/Downloads/nsfw_data_scrapper/raw_data/sexy/')