# English-Chinese translation example.

import pandas as pd
import numpy as np
import json
import jieba


def en_token(sq):
    return " ".join(sq.strip().split(' '))


def zh_token(sq):
    return " ".join(jieba.lcut(sq.strip()))


def split(N, tags=('train', 'val', 'test'), ps=(0.7, 0.2, 0.1)):
    return np.random.choice(tags, N, ps)


def data_process(trans_json):
    lines = []
    with open(trans_json, 'r') as fr:
        for i, line in enumerate(fr.readlines()):
            if i % 1001 == 1: 
                print('process: ', i)
            ct = json.loads(line.strip())
            lines.append((en_token(ct['english']), zh_token(ct['chinese'])))
    df = pd.DataFrame(lines, columns=['source_language', 'target_language'])
    df['split'] = split(len(lines))
    return df


def test_data_process():
    trans_json = "/home/liuxd/code4model/data/translation2019zh_train.json"
    save_path = "/home/liuxd/code4model/data/translation2019zh_train-df.csv"
    rtn = data_process(trans_json)
    rtn.to_csv(save_path, sep='\t', index=False)
    print(rtn)


if __name__ == "__main__":
    test_data_process()
