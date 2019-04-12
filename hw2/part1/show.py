import pickle
import numpy as np
object_1 = pickle.load(open('video_feat_dict.p', 'rb'))
object_2 = pickle.load(open('video_caption_dict.p', 'rb'))
object_3 = pickle.load(open('video_IDs.p', 'rb'))

one_keys = list(object_1.keys())
two_keys = list(object_2.keys())

one_values = list(object_1.values())
two_values = list(object_2.values())
print('video_feat example=', one_keys[:1], ' | ', one_values[:1])
print((one_values[0]).shape)
print('video_caption_dict example=', two_keys[:1], ' | ', two_values[:1])
print('video_IDs example=', object_3[:1])
