from glob import glob
from .base_dataset import BaseDataset
import random
import os
import pandas as pd
import io
from PIL import Image


class CC3MDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.metadata = None
        self._load_metadata()
        if split == "train":
            names = ["cc3m_train"]
        elif split == "val":
            names = ["cc3m_val"]
        elif split == "test":
            names = ["cc3m_val"]

        print(names, ": ", len(self.metadata), "samples in total.")
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        # self.data_dir = os.path.join(self.data_dir, 'cc3m')

    def _load_metadata(self):
        # download specific
        metadata_dir = './meta_data/cc3m'
        split_files = {
            'train': 'cc3m_training_success_full.tsv',
            'val': 'cc3m_validation_success_full.tsv',            # there is no test
            'test': 'cc3m_validation_success_full.tsv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        # elif self.split == 'val':
        #     metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        self.metadata = metadata

    def _get_image_path(self, sample):
        # conceptual captions uses this hashing to create the filename
        rel_dir = 'training'
        if self.split != 'train':
            rel_dir = 'validation'
        rel_fp = os.path.join(rel_dir, sample[1])
        #rel_fp = os.path.join(rel_dir, str(zlib.crc32(sample['thumbnailUrl'].encode('utf-8')) & 0xffffffff))
        # print(rel_fp)
        return os.path.join(self.data_dir, rel_fp), rel_fp

    def _get_caption(self, sample):
        return sample[0]
        #return sample['caption']

    def get_raw_image(self, sample):
        # print(sample)
        abs_fp, rel_fp = self._get_image_path(sample)
        img = Image.open(abs_fp).convert("RGB")
        if img is None:
            raise Exception("Invalid img!", rel_fp)
        else:
            return img

    def get_image(self, index, sample, image_key="image"):
        image = self.get_raw_image(sample)
        # image_tensor = [tr(image).unsqueeze(0) for tr in global_transforms]
        image_tensor = self.image_aug(image, self.transforms)
        # image_tensor[0] = image_tensor[0].unsqueeze(0)
        # print(len(image_tensor))
        # print(image_tensor[0].size())
        return {
            "video": image_tensor,
            "vid_index": sample[1],
            "cap_index": index,
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        image = self.get_raw_image(sample)
        # image_tensor = [tr(image).unsqueeze(0) for tr in global_transforms]
        image_tensor = self.image_aug(image, self.transforms)
        return {f"false_video_{rep}": image_tensor}

    def get_text(self, raw_index, sample):
        text = sample[0]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        # print(encoding.size())
        return {
            "text": (text, encoding),
            "vid_index": sample[1],
            "cap_index": raw_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        text = sample[0]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        # print(self.draw_false_image) # 1
        while result is None:
            sample = self.metadata.iloc[index]
            # print(sample)
            # try:
            ret = dict()
            ret.update(self.get_image(index, sample))
            if not self.image_only:
                txt = self.get_text(index, sample)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)

            for i in range(self.draw_false_image):
                ret.update(self.get_false_image(i))
            for i in range(self.draw_false_text):
                ret.update(self.get_false_text(i))
            result = True
            # except Exception as e:
            #     print(f"Error while read file idx {sample[1]} in {self.names[0]} -> {e}")
            #     index = random.randint(0, len(self.metadata) - 1)
            # ret["image"] = ret["image"].unsqueeze(1)
        return ret

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.get_suite(index)

