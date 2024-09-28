使用说明
1、修改训练脚本的训练数据路径。如bisai.py中第343行data_root改成数据集与标注的根目录，并且修改344行ann_file为标注pk相对data_root的路径。修改bisai.py最后一行work_dir为本地保存路径。
2、利用csv2pk.py生成ann_file。ann_file为一个dict的pickle保存结果，每个key是每个输入图片的绝对路径，对应的value也是一个dict: {'h': 图像高度, 'w': 图像宽度, 'b': 篡改框标注的numpy array且type为int64}。其中篡改框的标注形状为(n, 5)，其中n为篡改文本的数量，第2维度5的前4个是篡改框坐标最后一个固定为0，形式是(x_min, y_min, x_max, ymax, 0)。
3、训练命令是bash tools/dist_train.sh [配置文件] [显卡数量]

Instructions
1, Modify the training data path of the training script. For example, in line 343 of bisai.py, change data_root to the root directory of the dataset and annotation, and modify ann_file in line 344 to be the path of annotation pk relative to data_root. Modify the last line of work_dir in bisai.py to be the local save path.
2, Generate ann_file with csv2pk.py. ann_file is a dict of pickle save results, each key is the absolute path of each input image, the corresponding value is also a dict: {‘h’: image height, ‘w’: image width, ‘b’: tampering with the box labelled numpy array and type int64}. where the tamper box is labelled with shape (n, 5).
Where n is the number of tampered with text, the first 4 of the 2nd dimension 5 are tampered with box coordinates the last one is fixed to 0, the form is (x_min, y_min, x_max, ymax, 0).
3, The training command is bash tools/dist_train.sh [configuration file] [number of GPUs]

