import pandas as pd
import sys
from Extraction_of_features import runner
import time
import multiprocessing as mp
import tqdm
import math
from functools import partial

inputpath = '/home/data4T/wym/fsl/traids/dataset/id2sourcecode/'

def get_sim(tool,dataframe):
    sim = []
    for _,pair in dataframe.iterrows():
        id1,id2 = pair.FunID1,pair.FunID2
        if tool[0] == 'c':               # CFG算法，且目标文件无法解析
            if str(id1) in wrongfile or str(id2) in wrongfile:
                sim.append('parse_errpr')
                continue
        sourcefile1 = inputpath + str(id1) + '.java'
        sourcefile2 = inputpath + str(id2) + '.java'
        try:
            # if tool == 'p1':
            #     if id1 == 11673906 and id2 == 11673911:
            #         similarity = 0.9547432550043516
            similarity = runner(tool, sourcefile1, sourcefile2)
        except Exception as e:
            similarity = repr(e).split('(')[0]
            log ="\n" + time.asctime() + "\t"+ tool + "\t" + str(id1) + "\t"+ str(id2) + "\t"+ similarity
            logfile.writelines(log)
            similarity = 'False'
        print(similarity)
        sim.append(similarity)

    return sim


def cut_df(df, n):
    df_num = len(df)
    every_epoch_num = math.floor((df_num/n))
    df_split = []
    for index in range(n):
        if index < n-1:
            df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
        else:
            df_tem = df[every_epoch_num * index:]
        df_split.append(df_tem)
    return df_split


def main():
    inputcsv = "/home/data4T/wym/fsl/precision/BCBcsv_onlyid/noclone.csv"
    #inputcsv = "/home/data4T/wym/fsl/precision/BCBcsv_onlyid/type-1.csv"
    Clonetype = inputcsv.split('/')[-1].split('.')[0]
    if 'noclone' in inputcsv:
        Clonetype = 'noclone'

    #methodtype = 't1'
    pairs = pd.read_csv(inputcsv, header=None)
    pairs = pairs.drop(labels=0)
    pairs.columns = ['FunID1','FunID2']

    # df_split = cut_df(pairs, 60)
    #
    # func = partial(get_sim, 'c3')
    # pool = mp.Pool(processes=60)
    # sim_c3 = []
    # it_sim_c3 = tqdm.tqdm(pool.imap(func, df_split))
    # for item in it_sim_c3:
    #     sim_c3 = sim_c3 + item
    # pool.close()
    # pool.join()

    sim_t1 = get_sim('t1', pairs)
    sim_t2 = get_sim('t2', pairs)
    sim_t3 = get_sim('t3', pairs)
    sim_a1 = get_sim('a1', pairs)
    sim_a2 = get_sim('a2', pairs)
    sim_a3 = get_sim('a3', pairs)
    sim_c1 = get_sim('c1', pairs)
    sim_c2 = get_sim('c2', pairs)
    sim_c3 = get_sim('c3', pairs)

    print(sim_t1, sim_t2, sim_t3, sim_a1, sim_a2, sim_a3, sim_c1, sim_c2, sim_c3)

    result = pd.DataFrame({'FunID1': pairs['FunID1'].to_list(), 'FunID2': pairs['FunID2'].to_list(),
                           't1_sim': sim_t1, 't2_sim': sim_t2, 't3_sim': sim_t3, 'a1_sim': sim_a1, 'a2_sim': sim_a2,
                           'a3_sim': sim_a3, 'c1_sim': sim_c1, 'c2_sim': sim_c2, 'c3_sim': sim_c3})

    result.to_csv('./output/' + Clonetype + '_sim.csv', index=False)


if __name__ == '__main__':
    parse_er_file = open('parser_error.txt', 'r')  # 没有则创建
    wrongfile = parse_er_file.read().split(' ')  # 解析失败的文件列表名不包含.java
    logfile = open('errorlog.txt','a')

    main()

    logfile.close()
