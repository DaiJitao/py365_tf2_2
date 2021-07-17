# encoding=utf8
import sys

# 连续的特征；
numic_feat = ['view_count', 'like_count', 'reply_count', 'quality_score', 'rank_score_a', 'nlp_content_score',
              'nlp_cover_score', 'nlp_effect_score', 'nlp_quality_score', 'nlp_rare_score', 'nlp_time_score',
              'base_weight', 'title_query_sim']
# 多值离散型
multi_value_cate_feat = ['query_keyword', 'title', 'car_brand_ids', 'cms_series_ids', 'cms_spec_ids']

t = 'in_{} = Input(shape=[{}_size], name="{}") # None*size, 最长长度{}.size'


if __name__ == '__main__':
    temp = '{}_Kd = RepeatVector(1)(Dense(latent)(in_{}))  # None * 1 * K'
    m = '{}_Kd'
    in_ = "in_{}"
    arrs = []
    for feat in multi_value_cate_feat:
        temp = t.format(feat, feat, feat, feat)
        print(temp)

    print(arrs)
