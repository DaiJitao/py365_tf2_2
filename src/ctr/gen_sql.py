# encoding=utf8
import sys


def gen_dict_sql(key):
    tb = 'ds_rds.search_feas_item_property_feature'
    
    p1 = "insert overwrite directory '/team/ds/daijitao/ctr/deepfm/featdicts/{}'".format(key)
    p2 = "ROW FORMAT DELIMITED FIELDS TERMINATED BY '\\t'"
    sql = "select distinct {} from {} where dt='2021-06-23';".format(key, tb)
    print(p1)
    print(p2)
    print(sql)
    print('\n\n')
    
if __name__ == '__main__1':
    key = "doc_id"
    gen_dict_sql(key)
    inf = 'feat_dict'
    with open(inf, mode='r', encoding='utf-8') as fp:
        for line in fp:
            arr = line.strip().split('\t')
            if arr:
                key = arr[0].strip()
                gen_dict_sql(key)
    

tb = 'dmf_fdm.dmf_fdm_fetr_rcmd_item_di_v2_search'
tb = 'ds_rds.search_feas_item_property_feature'
if __name__ == '__main__':
    sql = "select count({0}) from  {1} where dt='2021-06-23' and {0} is not null and {0} !='';"
    key = 'answer_count'
    t = sql.format(key, tb)
    print(t)
