from torch.utils.data import Dataset
import os
import cv2
import torchvision.transforms as transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MyDataSet(Dataset):
    """
    This Dataset only work for a folder that contains one class image!!!
    """

    def __init__(self, file_path):
        if not os.path.isdir(file_path):
            raise ValueError("input file_path is not a dir")
        self.file_path = file_path
        self.image_list = os.listdir(file_path)

        self.transforms = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, index):
        path = os.listdir(self.file_path)
        # 这行代码使用了Python中的os模块，并调用了listdir函数来获取指定路径下的所有文件和文件夹的名称列表。
        rpath = path[0]
        gpath = path[1]
        rimage_list = os.listdir(os.path.join(self.file_path, rpath))
        gimage_list = os.listdir(os.path.join(self.file_path, gpath))
        rimage_path = os.path.join(self.file_path, rpath, rimage_list[index])
        gimage_path = os.path.join(self.file_path, gpath, gimage_list[index])
        rimage = self._read_convert_image(rimage_path)
        gimage = self._read_convert_image(gimage_path)

        return rimage, gimage

    def _read_convert_image(self, image_name):
        image = cv2.imread(image_name)
        image = cv2.resize(image, (256, 256))
        image = self.transforms(image).float()
        return image
    # 这个函数接受一个图像文件路径作为输入，然后使用OpenCV库读取、调整大小，并通过一些预定义的转换进行最终的处理，最后返回处理后的图像。

    def _read_convert_images(self, image_name):
        image = cv2.imread(image_name)
        image = cv2.resize(image, (128, 128))
        image = self.transforms(image).float()
        return image

    def __len__(self):
        path = os.listdir(self.file_path)
        rpath = path[0]
        rimage_list = os.listdir(os.path.join(self.file_path, rpath))
        return len(rimage_list)
