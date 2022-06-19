import json
from typing import Any
import io
#import itertools

if __name__ == '__main__':
    print('start')
    conf_file: io.TextIOWrapper = open('utils/config/config.json')
    conf: dict = json.load(conf_file)
    for x in conf:
        print(x, conf[x])


#conf_parsed: List = split_conf(conf)
#for x in conf_parsed:
#    print(x['name'])

#def split_conf(conf): # in: modelling params with lists, out: list of params-combination
#    import itertools
#    result = []
#    for mtype in conf['models'].split(','):
#        conf_mtype = conf[mtype]
#        # if conf_mtype['tune']=='': return [conf_mtype]
#        if conf_mtype['tune']=='': result += [conf_mtype] #result.append([conf_mtype]) 
#        else:
#            model_configs = []; list_of_lists = []
#            def tunelists_to_floats(conf_mtype, comb):
#                conf_i = conf_mtype.copy()
#                for i,t in enumerate(conf_mtype['tune'].split(',')): conf_i[t] = comb[i]
#                return conf_i
#            # 1 build list of lists
#            for t in conf_mtype['tune'].split(','): list_of_lists.append(conf_mtype[t])
#            # 2 get all combinations of a list of lists
#            combs = list(itertools.product(*list_of_lists)) # [(2734, 10), (2734, 20), (2734, 30), (1, 10), (1, 20), (1, 30), (2, 10), (2, 20), (2, 30)]
#            # 3 fill separate model_config from each combination 
#            for comb in combs:model_configs.append(tunelists_to_floats(conf_mtype, comb))
#            # return model_configs
#            result += model_configs #result.append(model_configs)
#            #print(split_model_config(conf['lgb']))
    
#    return result
