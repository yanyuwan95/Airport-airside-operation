import pandas as pd
import numpy as np
from xml.etree import ElementTree
import configparser
from myFunction import *
import random
#pd.set_option('max_rows', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from itertools import product
class findDepartInfo(object):
    def __init__(self, control_config, random_seed, dict_rwy_id, df_highest_OD):
        np.random.seed(random_seed)
        self.arrivalControlOption = float(control_config['VariationSettings']['arrivalControl'])
        self.gatePushControlOption = float(control_config['VariationSettings']['getPushControl'])
        self.dict_rwy_id = dict_rwy_id # {1:'07L/25R', 2: '07R/25L',3: '06R/24L', 4: '06L/24R'}
        self.df_highest_OD = df_highest_OD


        lst_highest_OD_key = list(zip(self.df_highest_OD['rwy_id'].values.tolist(), self.df_highest_OD['ramp_id'].values.tolist(), self.df_highest_OD['operation'].values.tolist()))


        self.dict_highest_OD = dict(zip(lst_highest_OD_key, self.df_highest_OD['update_edge_seq'].values.tolist()))
        # {('06L/24R', '0', 'A'): '[1349, 1351]}'

    def rewrite_update_edge_seq(self, route_i):
        route_i = route_i.split('[')[1]
        route_i = route_i.split(']')[0]
        route_i = route_i.split(', ')
        route_i = ['E_' + x for x in route_i]
        route_i = ' '.join(route_i)
        return route_i

    def addScheduleVar(self, df):
        arrivalVar = 0
        departureVar = 0
        ### initial schedule control ###
        if self.arrivalControlOption==0 and self.gatePushControlOption==0:
            return df
        else:
            df['update_depart_time'] = df['depart_time'].values.tolist()
            df_A = df[df['operation'] == 'A']
            df_D = df[df['operation'] == 'D']
            arrivalVar = self.arrivalControlOption
            gateVar = self.gatePushControlOption
            lst_var_A = np.random.uniform(0, arrivalVar, size=len(df_A))
            lst_var_D = np.random.uniform(0, gateVar, size=len(df_D))
            if not self.arrivalControlOption == 0:
                df_A['var_depart_time'] = lst_var_A
                df_A['update_depart_time'] = df_A['depart_time'] + df_A['var_depart_time']
                df_A = df_A.drop(columns=['var_depart_time'])
            if not self.gatePushControlOption == 0:
                gateVar = self.gatePushControlOption
                df_D['var_depart_time'] = lst_var_D
                df_D['update_depart_time'] = df_D['depart_time'] + df_D['var_depart_time']
                df_D = df_D.drop(columns=['var_depart_time'])
            df = pd.concat([df_A, df_D])
            df = df.drop(columns=['depart_time'])
            df = df.rename(columns={'update_depart_time': 'depart_time'})
            return df

    def findDepartDictandDf(self, dict_state, df_traffic_scenario, dict_min_rwy_sep):
        # dict_state{air_id:rwy_ref}
        # df_traffic_scenario {'veh_id', 'operation', 'ramp_id', 'rwy_id', 'taxi_time', 'reg_num', 'call_sign', 'type', 'ICAO_type'}
        # dict_min_sep {(M, H): 64}
        # self.dict_highest_OD {}
        df_depart = df_traffic_scenario[['operation','veh_id', 'ramp_id', 'ICAO_type', 'fuel_rate_idle', 'fuel_rate_approach']]

        df_depart.loc[:, 'rwy_id'] = df_depart['veh_id'].map(dict_state)

        lst_rwy_ramp = list(zip(df_depart['rwy_id'].values.tolist(), df_depart['ramp_id'].values.tolist(), df_depart['operation'].values.tolist()))

        df_depart.loc[:, 'edge_list'] = [x if x not in self.dict_highest_OD else self.dict_highest_OD[x] for x in lst_rwy_ramp]

        df_depart.loc[:, 'depart_time'] = df_traffic_scenario['start_time_in_minute_update']

        # lst_df_depart = [g for _, g in df_depart.groupby('rwy_id')]
        #
        # for df_i in lst_df_depart:
        #     rwy_id = df_i['rwy_id'].iloc[0]
        #     for i in range(len(df_i) - 1):
        #         j = i + 1
        #         air_pre_id = df_i['veh_id'].iloc[i]
        #         air_next_id = df_i['veh_id'].iloc[j]
        #         air_pre_type = df_traffic_scenario[df_traffic_scenario['veh_id']==air_pre_id]['ICAO_type'].iloc[0]
        #         air_next_type = df_traffic_scenario[df_traffic_scenario['veh_id'] == air_next_id]['ICAO_type'].iloc[0]
        #         min_sep_time = dict_min_rwy_sep[(air_pre_type, air_next_type)]
        #         df_i['depart_time'].iloc[j] = df_i['depart_time'].iloc[i] + int(min_sep_time)



        # df_depart = pd.concat(lst_df_depart)

        df_depart.loc[:, 'first_depart_edge'] = df_depart['edge_list'].apply(lambda x: x.split(' ')[0])

        df_depart = self.addScheduleVar(df_depart)

        df_depart.loc[:, 'veh_id'] = df_depart['veh_id'].astype(str)


        dict_depart_info = {}
        for i in range(len(df_depart)):
            dict_depart_info[df_depart['veh_id'].iloc[i]] = {'edge_list': df_depart['edge_list'].iloc[i],
                                                             'operation': df_depart['operation'].iloc[i], 'first_depart_edge': df_depart['first_depart_edge'].iloc[i],
                                                             'rwy_id': df_depart['rwy_id'].iloc[i], 'ramp_id': df_depart['ramp_id'].iloc[i],
                                                             'depart_time': df_depart['depart_time'].iloc[i],
                                                             'ICAO_type':df_depart['ICAO_type'].iloc[i],
                                                             'fuel_rate_idle': df_depart['fuel_rate_idle'].iloc[i],
                                                             'fuel_rate_approach': df_depart['fuel_rate_approach'].iloc[i]}

        return dict_depart_info, df_depart

def create_min_sep_time():
    lst_air_type = ['L', 'M', 'H']
    lst_air_type_comb =  list(product(lst_air_type , repeat=2))
    dict_min_sep = {}
    for comb_i in lst_air_type_comb:
        if comb_i[0] == 'L' and comb_i[1] == 'L':
            dict_min_sep[comb_i] = 80
        if comb_i[0] == 'M' and comb_i[1] == 'L':
            dict_min_sep[comb_i] = 160
        if comb_i[0] == 'H' and comb_i[1] == 'L':
            dict_min_sep[comb_i] = 240
        if comb_i[0] == 'L' and comb_i[1] == 'M':
            dict_min_sep[comb_i] = 68
        if comb_i[0] == 'M' and comb_i[1] == 'M':
            dict_min_sep[comb_i] = 73
        if comb_i[0] == 'H' and comb_i[1] == 'M':
            dict_min_sep[comb_i] = 150
        if comb_i[0] == 'L' and comb_i[1] == 'H':
            dict_min_sep[comb_i] = 64
        if comb_i[0] == 'M' and comb_i[1] == 'H':
            dict_min_sep[comb_i] = 64
        if comb_i[0] == 'H' and comb_i[1] == 'H':
            dict_min_sep[comb_i] = 100
    dict_min_sep[('NA', 'L')] = 0
    dict_min_sep[('NA', 'M')] = 0
    dict_min_sep[('NA', 'H')] = 0
    return dict_min_sep
if __name__ == '__main__':
    control_config = configparser.ConfigParser()
    config = loadConfigForAirport('KLAX_test')
    trafficFileLoc = config['Path']['trafficInfoLoc']
    simFileLoc = config['Path']['simInfoLoc']+'highest_freq_traffic_demand/'
    control_config.read(simFileLoc + 'simulationSetting_1.ini')

    random_seed = 1
    dict_rwy_id = {1: '07L/25R', 2: '07R/25L', 3: '06R/24L', 4: '06L/24R'}
    df_highest_OD = pd.read_csv(simFileLoc + 'highest_freq_route.csv')
    a = findDepartInfo(control_config, random_seed, dict_rwy_id, df_highest_OD)

    df_traffic_scenario = pd.read_csv(trafficFileLoc + 'scenario_1.csv')
    df_traffic_scenario = df_traffic_scenario.rename(columns={'trajectory_id': 'veh_id'})
    df_traffic_scenario = df_traffic_scenario.sort_values(by=['veh_id'])


    lst_air_id = df_traffic_scenario['veh_id'].values.tolist()

    raw_state = [random.randint(1, len(dict_rwy_id)) for i in range(len(df_traffic_scenario))]
    state = [dict_rwy_id[x] for x in raw_state]

    dict_state = dict(zip(lst_air_id, state))

    dict_min_rwy_sep = create_min_sep_time()
    print(dict_min_rwy_sep)

    b, c = a.findDepartDictandDf(dict_state , df_traffic_scenario, dict_min_rwy_sep)

    print(c)