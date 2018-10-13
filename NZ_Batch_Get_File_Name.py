import os

fd = open('/home/yaoqiang/data/nz432.txt','w')

path = r'/home/yaoqiang/data/nz432'
for dirpath,dirnames,filenames in os.walk(path):
    for filename in filenames:
        # print(os.path.join(dirpath,filename))
        pic = os.path.join(dirpath,filename)
        fd.write(pic+' 1\n')

fd.close()

# fd = open('/home/yaoqiang/data/nz432.txt','r')
# for line in fd:
#     print(line)

with open('/home/yaoqiang/data/nz432.txt','r') as fd:
    while True:
        line = fd.readline()
        if not line:
            break
        print(line)






      