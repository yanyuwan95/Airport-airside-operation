import sys

import pandas as pd
import numpy as np
import pickle
from myFunction import *
from itertools import product
import osmnx as ox
import networkx as nx
from SimSA_v2.Graph_map_infor import Graph_map_infor

def add_hallucinate_route():
    config = loadConfigForAirport('KLAX_test')
    mapFileLoc = config['Path']['mapInfoLoc']
    simFileLoc = config['Path']['simInfoLoc']
    trafficFileLoc = config['Path']['trafficInfoLoc']

    df_ramp_node = pd.read_csv(mapFileLoc + 'manual_ramp_with_node.csv')
    lst_ramp_id = df_ramp_node['ramp_id'].values.tolist()
    # lst_ramp_id = [str(x) for x in lst_ramp_id]

    df_edge  =pd.read_csv(mapFileLoc + 'df_edge.csv')

    dict_map_info = pickle.load(open(mapFileLoc + 'dict_map_info.pkl', 'rb'))
    dict_rwy_to_edge = dict_map_info['dict_rwy_to_edge']
    lst_rwy_id = list(dict_rwy_to_edge.keys())

    G = ox.load_graphml(mapFileLoc + 'devide_KLAX_test.graphml')


    G_info = Graph_map_infor(G)
    G = G_info.convert_node_attr_types()
    G = G_info.convert_edge_attr_types()

    G_r = G


    df_rwy_exit = pd.read_csv(mapFileLoc + 'mannual_rwy_exit.csv')
    df_ramp_node = pd.read_csv(mapFileLoc + 'manual_ramp_with_node.csv')
    df_double_edge = pd.read_csv(mapFileLoc + 'double_edge.csv')
    df_double_edge['ref'] = df_double_edge['ref'].fillna(value='none')
    df_rwy_edge = df_double_edge[df_double_edge['ref'].str.contains('/')]
    df_rwy_edge = df_rwy_edge.drop_duplicates(subset=['edge_osmid'])
    print(G_r.number_of_edges())
    for i in range(len(df_rwy_edge)):
        u = df_rwy_edge['origin_u'].iloc[i]
        v = df_rwy_edge['origin_v'].iloc[i]
        try:
            G_r.remove_edge(u, v)
        except:
            G_r.remove_edge(v, u)
    print(G_r.number_of_edges())

    G_r = G_r.to_undirected()
    lst_operation = ['A', 'D']

    lst_comb_ramp_rwy = list(product(lst_ramp_id, lst_rwy_id, lst_operation))

    df_highest_OD = pd.read_csv(trafficFileLoc + 'highest_freq_route.csv')
    df_highest_OD = df_highest_OD[~(df_highest_OD['ramp_id']=='none')]
    df_highest_OD['ramp_id'] = df_highest_OD['ramp_id'].astype(int)



    df_update_highest_OD = pd.DataFrame()
    print(df_highest_OD.dtypes)

    lst_ramp = []
    lst_rwy = []
    lst_operation = []
    lst_update_edge = []
    for comb_i in lst_comb_ramp_rwy:
        ramp_id = comb_i[0]
        rwy_id = comb_i[1]
        operation = comb_i[2]
        print(comb_i)
        lst_ramp.append(ramp_id)
        lst_rwy.append(rwy_id)
        lst_operation.append(operation)

        try:
            loc_edge_list = df_highest_OD[(df_highest_OD['operation']==operation) & (df_highest_OD['rwy_id']==rwy_id) & (df_highest_OD['ramp_id']==ramp_id)]['update_edge_seq'].iloc[0]
            print('find******************************************')
        except:
            loc_edge_list = find_similar_OD(operation, ramp_id, rwy_id, df_highest_OD, G_r, df_rwy_exit, df_ramp_node, df_double_edge)

        lst_update_edge.append(loc_edge_list)

    df_update_highest_OD['rwy_id']= lst_rwy
    df_update_highest_OD['ramp_id'] = lst_ramp
    df_update_highest_OD['operation'] = lst_operation
    df_update_highest_OD['update_edge_seq'] = lst_update_edge

    df_update_highest_OD['update_edge_seq'] = df_update_highest_OD['update_edge_seq'].apply(rewrite_update_edge_seq)

    df_update_highest_OD.to_csv(trafficFileLoc+'update_highest_freq_OD.csv', index=False)

def rewrite_update_edge_seq(route_i):
    route_i = route_i.split('[')[1]
    route_i = route_i.split(']')[0]
    route_i = route_i.split(', ')
    route_i = ['E_' + x for x in route_i]
    route_i = ' '.join(route_i)
    return route_i



def find_similar_OD(operation, ramp_id, rwy_id, df_highest_OD, G_r, df_rwy_exit, df_ramp_node, df_double_edge):
    # dict_similar_rwy = {'06R/24L':'06L/24R', '06L/24R':'06R/24L', '07L/25R':'07R/25L', '07R/25L':'07L/25R'}
    # try loc the most freq exit or enter rwy loc
    if operation=='A':
        try:
            loc_rwy_node = df_rwy_exit[(df_rwy_exit['rwy_id'] == rwy_id) & (df_rwy_exit['operation']==operation)]['node_id'].iloc[0]
        except:
            print(rwy_id)
            print(type(rwy_id))
            sys.exit(1)
        loc_rwy_edge = df_rwy_exit[(df_rwy_exit['rwy_id'] == rwy_id) & (df_rwy_exit['operation'] == operation)]['rwy_edge_id'].iloc[0]
        try:
            rwy_osmid_node = df_double_edge[df_double_edge['node1_id']==loc_rwy_node]['origin_u'].iloc[0]
        except:
            print(loc_rwy_node)
            print(type(loc_rwy_node))
            print(df_double_edge.dtypes)
            sys.exit(1)

        ramp_node = df_ramp_node[df_ramp_node['ramp_id']==ramp_id]['node_id'].iloc[0]
        try:
            ramp_osmid_node = df_double_edge[df_double_edge['node1_id']==ramp_node]['origin_u'].iloc[0]
        except:
            print(ramp_node)
            print(type(ramp_node))
            sys.exit(1)


        lst_shortest_route = nx.shortest_path(G_r, source=rwy_osmid_node,
                                          target=ramp_osmid_node, weight='length')

        lst_update_edge = [loc_rwy_edge]
        for i in range(int(len(lst_shortest_route)-1)):
            edge_i = df_double_edge[(df_double_edge['origin_u']==lst_shortest_route[i]) & (df_double_edge['origin_v']==lst_shortest_route[int(i+1)])]['edge_id'].iloc[0]
            lst_update_edge.append(edge_i)
    if operation=='D':
        loc_rwy_node = df_rwy_exit[(df_rwy_exit['rwy_id'] == rwy_id) & (df_rwy_exit['operation'] == operation)]['node_id'].iloc[0]
        loc_rwy_edge = df_rwy_exit[(df_rwy_exit['rwy_id'] == rwy_id) & (df_rwy_exit['operation'] == operation)]['rwy_edge_id'].iloc[0]
        rwy_osmid_node = df_double_edge[df_double_edge['node1_id'] == loc_rwy_node]['origin_u'].iloc[0]

        ramp_node = df_ramp_node[df_ramp_node['ramp_id'] == ramp_id]['node_id'].iloc[0]
        ramp_osmid_node = df_double_edge[df_double_edge['node1_id'] == ramp_node]['origin_u'].iloc[0]

        lst_shortest_route = nx.shortest_path(G_r, source=ramp_osmid_node,
                                              target=rwy_osmid_node, weight='length')

        lst_update_edge = []
        for i in range(int(len(lst_shortest_route) - 1)):
            edge_i = df_double_edge[(df_double_edge['origin_u'] == lst_shortest_route[i]) & (
                        df_double_edge['origin_v'] == lst_shortest_route[int(i + 1)])]['edge_id'].iloc[0]
            lst_update_edge.append(edge_i)

        lst_update_edge.append(loc_rwy_edge)


    lst_update_edge = [str(x) for x in lst_update_edge]
    lst_update_edge = ', '.join(lst_update_edge)
    string_lst_update_edge = '['+lst_update_edge+']'
    return string_lst_update_edge

if __name__ == '__main__':
    add_hallucinate_route()

