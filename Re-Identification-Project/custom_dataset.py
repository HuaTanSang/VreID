import os, glob
import torchreid
from torchreid.data.datasets.dataset import ImageDataset

class MyDataset(ImageDataset):
    dataset_dir = 'dataset_0'

    def __init__(self, root='', **kwargs):
        # root = '/home/duypham/BigData_Project_ReID/Custom_ReID_Dataset'
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        train = self._process_dir(os.path.join(self.dataset_dir, 'bounding_box_train'), relabel=True)
        query = self._process_dir(os.path.join(self.dataset_dir, 'query'), relabel=False)
        gallery = self._process_dir(os.path.join(self.dataset_dir, 'gallery'), relabel=False)
        super(MyDataset, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pid_set = set(int(os.path.basename(p).split('_')[0]) for p in img_paths)
        pid2label = {pid: i for i, pid in enumerate(pid_set)}
        data = []
        for p in img_paths:
            fn = os.path.basename(p)
            pid = int(fn.split('_')[0])
            camid = int(fn.split('_')[1][1])  # 'c1' → 1
            if relabel:
                pid = pid2label[pid]
            data.append((p, pid, camid))
        return data

# Đăng ký vào TorchReID
torchreid.data.register_image_dataset('dataset_0', MyDataset)
