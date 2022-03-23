import os.path as op
from zipfile import ZipFile, BadZipFile
import torch.utils.data as data
from PIL import Image
from io import BytesIO
import multiprocessing

_VALID_IMAGE_TYPES = ['.jpg', '.jpeg', '.tiff', '.bmp', '.png']

class ZipData(data.Dataset):
    _IGNORE_ATTRS = {'_zip_file'}

    def __init__(self, path, map_file,
                 transform=None, target_transform=None,
                 extensions=None):
        self._path = path
        if not extensions:
            extensions = _VALID_IMAGE_TYPES
        self._zip_file = ZipFile(path)
        self.zip_dict = {}
        self.samples = []
        self.transform = transform
        self.target_transform = target_transform
        self.class_to_idx = {}
        with open(map_file, 'r') as f:
            for line in iter(f.readline, ""):
                line = line.strip()
                if not line:
                    continue
                cls_idx = [l for l in line.split('\t') if l]
                if not cls_idx:
                    continue
                assert len(cls_idx) >= 2, "invalid line: {}".format(line)
                idx = int(cls_idx[1])
                cls = cls_idx[0]
                del cls_idx
                at_idx = cls.find('@')
                assert at_idx >= 0, "invalid class: {}".format(cls)
                cls = cls[at_idx + 1:]
                if cls.startswith('/'):
                    # Python ZipFile expects no root
                    cls = cls[1:]
                assert cls, "invalid class in line {}".format(line)
                prev_idx = self.class_to_idx.get(cls)
                assert prev_idx is None or prev_idx == idx, "class: {} idx: {} previously had idx: {}".format(
                    cls, idx, prev_idx
                )
                self.class_to_idx[cls] = idx

        for fst in self._zip_file.infolist():
            fname = fst.filename
            target = self.class_to_idx.get(fname)
            if target is None:
                continue
            if fname.endswith('/') or fname.startswith('.') or fst.file_size == 0:
                continue
            ext = op.splitext(fname)[1].lower()
            if ext in extensions:
                self.samples.append((fname, target))
        assert len(self), "No images found in: {} with map: {}".format(self._path, map_file)

    def __repr__(self):
        return 'ZipData({}, size={})'.format(self._path, len(self))

    def __getstate__(self):
        return {
            key: val if key not in self._IGNORE_ATTRS else None
            for key, val in self.__dict__.iteritems()
        }

    def __getitem__(self, index):
        proc = multiprocessing.current_process()
        pid = proc.pid # get pid of this process.
        if pid not in self.zip_dict:
            self.zip_dict[pid] = ZipFile(self._path)
        zip_file = self.zip_dict[pid]

        if index >= len(self) or index < 0:
            raise KeyError("{} is invalid".format(index))
        path, target = self.samples[index]
        try:
            sample = Image.open(BytesIO(zip_file.read(path))).convert('RGB')
        except BadZipFile:
            print("bad zip file")
            return None, None
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)