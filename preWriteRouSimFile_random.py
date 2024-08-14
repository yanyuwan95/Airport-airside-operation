import pandas as pd
import numpy as np
from multiprocessing import Pool
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom
import random
import os
from myFunction import *
import shutil
'''
writeRouSimFile: write the Rou and SUMO configuraion file for simulation
input: sumo OD pairs, traffic scenario, and the route selection approach
output: Rou and SUMO file
'''

class writeRouSimFile(object):
    def __init__(self, airport, df_traffic_demand, traffic_demand_scenario, route_select):
        self.airport = airport


        ### load configuration
        config = loadConfigForAirport(self.airport)
        self.mapFileLoc = config['Path']['mapInfoLoc']
        self.traFileLoc = config['Path']['traInfoLoc']
        self.trafficFileLoc = config['Path']['trafficInfoLoc']
        self.simFileLoc = config['Path']['simInfoLoc']



        self.df_double_edge = pd.read_csv(self.mapFileLoc+'double_edge.csv')
        self.df_OD = pd.read_csv(self.trafficFileLoc+'OD_tra.csv')
        self.save_file_loc = self.simFileLoc+route_select+'/'
        createFolder(self.save_file_loc)

        if route_select == 'highest_freq_traffic_demand':
            self.write_STR_random(df_traffic_demand, traffic_demand_scenario)

    def write_STR_random(self, df_traffic_demand, traffic_demand_scenario):


        top_with_onlyTime = Element('routes')
        comment_with_onlyTime = Comment('only_depart_time_info')
        top_with_onlyTime.append(comment_with_onlyTime)


        vehicle_type_onlyTime = SubElement(top_with_onlyTime,'vType',
                                  attrib={'id': 'aircraft', 'accel': '9.8', 'decel': '9.8', 'sigma': '0.5',
                                          'length': '74', 'maxSpeed': '250', 'insertionChecks': 'none'} )

        standard_taxi_speed = str(5.68)  # generated from: Speed_Analyze/speed_analyze.py

        dict_operation_color = {'D': 'red', 'A': 'yellow'}
        for i in range(len(df_traffic_demand)):
            operation_i = df_traffic_demand['operation'][i]

            vehicle_only_departTime = SubElement(top_with_onlyTime, 'vehicle',
                                 attrib={'id': str(df_traffic_demand['trajectory_id'][i]),
                                         'type': 'aircraft',
                                         'color': dict_operation_color[operation_i],
                                         'depart': str(0),
                                         'departSpeed': str(0)})


            dict_initial_edge = {'D': 'E_10001', 'A':'E_10000'}
            route_only_departTime = SubElement(vehicle_only_departTime, 'route', attrib={'edges': dict_initial_edge[operation_i]})

        # write _with_only_departTime.rou.xml
        with open(self.save_file_loc + traffic_demand_scenario + '_with_only_departTime.rou.xml', 'w') as roufile:
            print(self.prettify(top_with_onlyTime), file=roufile)

        self.write_Sim(traffic_demand_scenario)


    def write_Sim(self, scenario):
        save_sumocfg_file_name = self.save_file_loc +'/'+str(scenario)+'.sumocfg'
        write_route_file_name = str(scenario)+'_with_only_departTime.rou.xml'

        top = Element('configuration', attrib={"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                                               "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/sumoConfiguration.xsd"})
        input_att = SubElement(top, 'input')
        net_file = SubElement(input_att, 'net-file', attrib={"value": self.airport+".net.xml"})

        route_file = SubElement(input_att, 'route-files', attrib={"value": write_route_file_name})

        ignore_route_errors = SubElement(input_att, 'ignore-route-errors', attrib={"value": "true"})

        collision_action = SubElement(input_att, 'collision.action', attrib={"value": "none"})

        teleport_control = SubElement(input_att, 'time-to-teleport', attrib={"value": "-1"})

        eager_insert = SubElement(input_att, 'eager-insert', attrib={"value": "true"})

        with open(save_sumocfg_file_name, 'w') as sumofile:
            print(self.prettify(top), file=sumofile)


    def prettify(self, elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="    ")

if __name__ == '__main__':
    config = loadConfigForAirport('KLAX_test')
    trafficFileLoc = config['Path']['trafficInfoLoc']
    df_traffic_demand = pd.read_csv(trafficFileLoc +'scenario_1.csv')
    _writeRouFile = writeRouSimFile(airport='KLAX_test', df_traffic_demand=df_traffic_demand, traffic_demand_scenario='1', route_select='highest_freq_traffic_demand')