# 将1-600、2-600等文件夹里面的图片全部合并，生成id-class对应的csv文件
import os
import pandas as pd

class_folder_path = ['./复杂']
img_to_class = {}

# 循环读取文件夹，生成{"img_id":class_num}格式的字典
for paths in enumerate(class_folder_path):
    index,path = paths
    imgs_name = os.listdir(path)
    for imgs in enumerate(imgs_name):
        _,img = imgs
        img_to_class[img] = 'unknow'

# 把字典转成dataframe格式
df = pd.DataFrame.from_dict(img_to_class,orient='index',columns=['class'])
df = df.reset_index().rename(columns={"index":"img_id"})

# 生成csv文件
df.to_csv("testset_fuza.csv",index=False,encoding="gb2312")

print("Done")