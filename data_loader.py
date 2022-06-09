import utils
import tagging_converter
import tagging
import numpy as np
from sklearn.model_selection import train_test_split
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer

from tqdm import tqdm

max_len = 256
batch_size = 4
label_map_file = 'data/csl_10k/label_map.txt'
input_file = 'data/csl_10k/train.tsv'
input_format = 'csl'
dict_path = r'D:\PekingInfoResearch\pretrain_models\chinese_L-12_H-768_A-12/vocab.txt'

label_map = utils.read_label_map(label_map_file)
_id_2_tag = {tag_id: tagging.Tag(tag) for tag, tag_id in label_map.items()}
converter = tagging_converter.TaggingConverter(
      tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
      False)
D = []
ignored = total = 0
for i, (sources, target) in tqdm(enumerate(utils.yield_sources_and_targets(input_file, input_format))):
    task = tagging.EditingTask(sources)
    tags = converter.compute_tags(task, target)
    if not tags:
        ignored += 1
        converter.compute_tags(task, target)
        continue
    total += 1
    D.append((sources, target, tags))
    # if i > 1000:
    #     break
print(f'coverage: {(total-ignored)/total}')

train_data, dev_data = train_test_split(D, test_size=0.1, random_state=42, shuffle=True)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    def __iter__(self, random=False):
        cur_ids, cur_seg, cur_label, cur_mask = [], [], [], []
        for is_end, d in self.sample(random):
            if not d[2]:
                continue
            labels = [label_map[str(tag)] for tag in d[2][:max_len-2]]
            labels = [0] + labels + [0]

            input_ids = tokenizer.tokens_to_ids(['[CLS]']+d[0][0].split()[:max_len-2]+['[SEP]'])
            segment_ids = [0] * len(input_ids)
            attention_mask = [1] * len(input_ids)
            cur_ids.append(input_ids)
            cur_seg.append(segment_ids)
            cur_label.append(labels)
            cur_mask.append(attention_mask)
            if len(cur_ids) == self.batch_size or is_end:
                cur_ids = sequence_padding(cur_ids)
                cur_seg = sequence_padding(cur_seg)
                cur_label = sequence_padding(cur_label)
                cur_mask = sequence_padding(cur_mask)
                cur_label = np.expand_dims(cur_label, -1)
                yield [cur_ids, cur_seg], cur_label
                cur_ids, cur_seg, cur_label, cur_mask = [], [], [], []

train_loader = data_generator(train_data, batch_size=batch_size)
dev_loader = data_generator(dev_data, batch_size=batch_size)
# for batch in train_loader.forfit():
#     pass