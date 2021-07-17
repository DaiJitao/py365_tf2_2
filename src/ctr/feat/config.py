# encoding=utf8
import sys
import yaml

dict_feature = {
    'answer_count':'dense', 'area_id': 'sparse'
}

inf = 'config.yaml'
with open(inf, encoding='utf-8', mode='r') as fp:
    config = yaml.load(fp, Loader=yaml.SafeLoader)
    print(config)