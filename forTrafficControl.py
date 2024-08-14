import os
import sys
import optparse
import traci
import numpy as np
import pandas as pd
from sumolib import checkBinary
import itertools
import osmnx as ox
import networkx as nx
import time
import datetime
from collections import Counter
from xml.etree import ElementTree
from tqdm import tqdm
import configparser
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from SimSA_v2.forSpeedControl import speedControl


class trafficControl(object):
    def __init__(self, dict_map_info, control_config, dict_min_rwy_sep, random_seed):
        self.random_seed = random_seed

        ### load map info ###
        self.dict_mirror_edge = dict_map_info['dict_mirror_edge']
        self.dict_rwy_to_edge = dict_map_info['dict_rwy_to_edge']
        self.dict_rwy_to_inter_edge = dict_map_info['dict_rwy_to_inter_edge']
        self.lst_rwy_to_inter_edge =[item for sublist in [x for x in self.dict_rwy_to_inter_edge.values()] for item in sublist]

        self.dict_rwy_edge_and_inter_edge = dict_map_info['dict_rwy_edge_and_inter_edge']

        self.lst_rwy_to_rwy_and_inter_edge = [item for sublist in [x for x in self.dict_rwy_edge_and_inter_edge.values()] for item in sublist]


        self.dict_nearby_edge = dict_map_info['dict_nearby_edge']
        self.dict_edge_to_node = dict_map_info['dict_edge_to_node']
        # self.dict_depart = dict_depart # {'veh_id':{'edge_list':$,'operation':$,'first_depart_edge':$,'rwy_id':$,'ramp_id':$, 'depart_time': $}}
        # self.df_depart = df_depart

        ### load control config###
        '''traffic rule'''
        self.if_rwy_separation = control_config['TrafficRule']['if_rwy_separation']
        self.if_taxi_separation = control_config['TrafficRule']['if_taxi_separation']
        self.if_rwy_incursion = control_config['TrafficRule']['if_rwy_incursion']
        '''specific traffic rule value'''
        self.col_dist = int(control_config['Separation']['col_dist'] if self.if_taxi_separation else 0)
        self.free_rwy_time_req = int(control_config['Separation']['rwy_sep'] if self.if_rwy_separation else 0)
        self.dict_rwy_free_time = {}
        self.dict_rwy_last_air = {}
        self.dict_min_rwy_seq = dict_min_rwy_sep

        '''specific variation parameter'''
        self.speedControlOption = control_config['VariationSettings']['speedControl']
        self.landingSpeedControlOption = control_config['VariationSettings']['landingSpeedControl']
        self.takeoffSpeedControlOption = control_config['VariationSettings']['takeoffSpeedControl']

        self.arrivalControlOption = float(control_config['VariationSettings']['arrivalControl'])
        self.gatePushControlOption = float(control_config['VariationSettings']['getPushControl'])

        ### initial speed control ###
        if self.speedControlOption == 'constantAvgSpeed':
            self._speedControl = speedControl(dict_map_info=dict_map_info, dict_option={'constantAvgSpeed': 5.68}, random_seed= self.random_seed)
        elif self.speedControlOption == 'randomSpeed':
            self._speedControl = speedControl(dict_map_info=dict_map_info, dict_option={'randomSpeed': {'lower':0, 'upper':12}}, random_seed= self.random_seed)
        elif self.speedControlOption == 'varOverAvgTaxiSpeed':
            self._speedControl = speedControl(dict_map_info=dict_map_info, dict_option={'varOverAvgTaxiSpeed': {'variation': 1}}, random_seed= self.random_seed)

        ### initial global vehicle location and update location###
        self.dict_vehicle = {}
        self.lst_cur_road_occupied = []
        self.dict_vehicle_control = {}

    def controlAtStep(self, dict_rwy_free_time, dict_rwy_last_air, dict_vehicle, step, df_depart, dict_depart):
        '''return a dict_vehicle_control {'veh_id': {'speed':, 'road_id':, 'edge_list':, 'if_hold':,}}'''
        ### 0 for no hold, 1 for hold land, 2 for hold gate, 3 for hold inter for takoff, 4 for hold inter for cross, 5 for park###
        self.dict_vehicle_control = {}

        '''load current traffic information'''
        self.dict_vehicle = dict_vehicle
        self.dict_rwy_free_time = dict_rwy_free_time
        self.dict_rwy_last_air = dict_rwy_last_air
        self.lst_cur_road_occupied = [x for x in dict_vehicle.values()]

        # control this step's aircraft
        for veh_id, veh_value in dict_vehicle.items():
            self.identify_control(veh_id, veh_value, step, df_depart, dict_depart)

        # count rwy free time
        lst_actual_occupied_road = self.lst_cur_road_occupied
        for rwy_id, rwy_edge in self.dict_rwy_to_edge.items():
            if any(item in lst_actual_occupied_road for item in rwy_edge):
                dict_rwy_free_time[rwy_id] = 0

        return self.dict_vehicle_control, dict_rwy_free_time, dict_rwy_last_air

    def identify_control(self, veh_id, veh_value, step, df_depart, dict_depart):
        # current location
        road_id = veh_value

        # scheduled info
        operation_id = dict_depart[veh_id]['operation']
        rwy_id = dict_depart[veh_id]['rwy_id']
        ramp_id = dict_depart[veh_id]['ramp_id']


        self.dict_vehicle_control[veh_id] = {'speed': 'none', 'road_id':road_id, 'edge_list': 'none', 'if_hold': 0}

        if not any([road_id == 'E_10001', road_id == 'E_10000']):
            if road_id in self.lst_rwy_to_inter_edge:
                if (operation_id == 'D') and (road_id in self.dict_rwy_to_inter_edge[rwy_id]):
                    self.runway_takeoff_control(veh_id, road_id, operation_id, rwy_id, ramp_id, dict_depart)
                else:
                    self.runway_incursion_control(veh_id, road_id, operation_id, rwy_id, ramp_id)
            else:
                self.taxiway_separation_control(veh_id, road_id, operation_id, rwy_id, ramp_id)
        else:
            self.check_reset_depart(veh_id, road_id, operation_id, rwy_id, ramp_id, step, df_depart, dict_depart)

    def check_if_towards(self, lst_intersect, road_id):
        # first find the lst_remove_self_cur_road_occupied
        if_towards = False
        road_id_node2 = self.dict_edge_to_node[road_id][1]
        for edge_i in lst_intersect:
            edge_node2 = self.dict_edge_to_node[edge_i][1]
            if edge_node2 == road_id_node2:
                if_towards = True
        return if_towards

    def taxiway_separation_control(self, veh_id, road_id, operation_id, rwy_id, ramp_id):
        if road_id in self.lst_rwy_to_rwy_and_inter_edge:
            speed_i = self._speedControl.return_speed(road_id)
        else:
            lst_nearby_edge = self.dict_nearby_edge[road_id]
            lst_remove_self_cur_road_occupied = [x for x in self.lst_cur_road_occupied if x!=road_id]
            lst_intersect = list(set(lst_remove_self_cur_road_occupied).intersection(set(lst_nearby_edge)))
            if len(lst_intersect)>0:
            # if any(item in lst_remove_self_cur_road_occupied for item in lst_nearby_edge):
                checkIfTowards = self.check_if_towards(lst_intersect, road_id)
                if checkIfTowards:
                    speed_i = 0
                    self.lst_cur_road_occupied.remove(road_id)
                    self.dict_vehicle_control[veh_id]['if_hold'] = 5
                else:
                    speed_i = self._speedControl.return_speed(road_id)
            else:
                speed_i = self._speedControl.return_speed(road_id)
        self.dict_vehicle_control[veh_id]['speed'] = speed_i

    def runway_takeoff_control(self, veh_id, road_id, operation_id, rwy_id, ramp_id, dict_depart):
        last_air_at_rwy = self.dict_rwy_last_air[rwy_id]
        cur_air_type =dict_depart[veh_id]['ICAO_type']
        need_seq = self.dict_min_rwy_seq[(last_air_at_rwy, cur_air_type)]
        if self.dict_rwy_free_time[rwy_id] > need_seq:
            speed_i = self._speedControl.return_speed(road_id)
            self.dict_rwy_last_air[rwy_id] = cur_air_type
        else:
            speed_i = 0
            self.dict_vehicle_control[veh_id]['if_hold'] = 3
        self.dict_vehicle_control[veh_id]['speed'] = speed_i

    def runway_incursion_control(self, veh_id, road_id, operation_id, rwy_id, ramp_id):
        if road_id in self.dict_rwy_to_inter_edge[rwy_id]:
            speed_i = self._speedControl.return_speed(road_id)
        else:
            for rwy_id, rwy_edge in self.dict_rwy_to_inter_edge.items():
                if road_id in rwy_edge:
                    rwy_ref = rwy_id
                    break
            if any(item in self.lst_cur_road_occupied for item in self.dict_rwy_to_edge[rwy_ref]):
                speed_i = 0
                self.dict_vehicle_control[veh_id]['if_hold'] = 4
            else:
                speed_i = self._speedControl.return_speed(road_id)
        self.dict_vehicle_control[veh_id]['speed'] = speed_i

    def check_reset_depart(self, veh_id, road_id, operation_id, rwy_id, ramp_id, step, df_depart, dict_depart):
        if all([self.arrivalControlOption > 0, operation_id == 'A']):
            if dict_depart[veh_id]['depart_time'] > step:
                self.dict_vehicle_control[veh_id]['speed'] = 0
            else:
                self.check_reset_depart_airport_regulation(veh_id, road_id, operation_id, rwy_id, ramp_id, df_depart, dict_depart, step)
        elif all([self.gatePushControlOption > 0, operation_id == 'D']):
            if dict_depart[veh_id]['depart_time'] > step:
                self.dict_vehicle_control[veh_id]['speed'] = 0
            else:
                self.check_reset_depart_airport_regulation(veh_id, road_id, operation_id, rwy_id, ramp_id, df_depart, dict_depart, step)
        else:
            self.check_reset_depart_airport_regulation(veh_id, road_id, operation_id, rwy_id, ramp_id, df_depart, dict_depart, step)

    # E_10001 for gate, E_10000 for land
    def check_reset_depart_airport_regulation(self, veh_id, road_id, operation_id, rwy_id, ramp_id, df_depart, dict_depart, step):
        if operation_id == 'A':
            last_air_at_rwy = self.dict_rwy_last_air[rwy_id]
            cur_air_type = dict_depart[veh_id]['ICAO_type']
            need_seq = self.dict_min_rwy_seq[(last_air_at_rwy, cur_air_type)]
            if self.dict_rwy_free_time[rwy_id] > need_seq:
                speed_i = self._speedControl.return_speed(road_id)
                edge_list_i = dict_depart[veh_id]['edge_list'].split(' ')
                self.dict_rwy_last_air[rwy_id] = cur_air_type
            else:
                speed_i = 0
                edge_list_i = ['E_10000']
                if dict_depart[veh_id]['depart_time'] > step:
                    self.dict_vehicle_control[veh_id]['if_hold'] = 1
            self.dict_vehicle_control[veh_id]['speed'] = speed_i
            self.dict_vehicle_control[veh_id]['edge_list'] = edge_list_i
        elif operation_id == 'D':
            lst_check_edge = self.dict_nearby_edge[dict_depart[veh_id]['first_depart_edge']]
            check_occupied = any(item in self.lst_cur_road_occupied for item in lst_check_edge)
            # if no aircraft in the nearby of the first depart edge
            if not check_occupied:
                speed_i = self._speedControl.return_speed(road_id)
                edge_list_i = dict_depart[veh_id]['edge_list'].split(' ')
            else:
                speed_i = 0
                edge_list_i = ['E_10001']
                if dict_depart[veh_id]['depart_time'] > step:
                    self.dict_vehicle_control[veh_id]['if_hold'] = 2
            self.dict_vehicle_control[veh_id]['speed'] = speed_i
            self.dict_vehicle_control[veh_id]['edge_list'] = edge_list_i