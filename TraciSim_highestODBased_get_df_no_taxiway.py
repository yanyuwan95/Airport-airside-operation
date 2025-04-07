import sys
import traci
from sumolib import checkBinary
import time
import configparser
import pickle
from SimSA_v3.forFindDepartInfo import findDepartInfo
from SimSA_v3.forTrafficControl import trafficControl
from SimSA_v3.forFindDepartInfo import create_min_sep_time

from SimSA_v3.myFunction import *
import random
from itertools import groupby

pd.set_option('max_rows', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class highestODSim(object):
    def __init__(self, airport, sumo_no_gui, option, sim_file_id, sumoFilename, random_seed, dict_min_rwy_sep, dict_rwy_id, df_traffic_scenario):
        self.random_seed =random_seed

        """load prepared file"""
        foldConfig = loadConfigForAirport(airport)
        self.simFileLoc = foldConfig['Path']['simInfoLoc']+option+'/'
        self.mapFileLoc = foldConfig['Path']['mapInfoLoc']
        self.trafficFileLoc = foldConfig['Path']['trafficInfoLoc']

        """set traffic rule parameter"""
        self.control_config = configparser.ConfigParser()
        self.control_config.read(self.simFileLoc + 'simulationSetting_' + str(sim_file_id) + '.ini')

        '''load map info'''
        self.dict_map_info = pickle.load(open(self.mapFileLoc+'dict_map_info.pkl', 'rb'))
        self.dict_operation = self.dict_map_info['dict_operation']

        """sumo simulation configuration"""
        # whether to use the GUI
        self.sumoBinary = checkBinary('sumo') if sumo_no_gui else checkBinary('sumo-gui')

        '''load corresponding sumo file'''
        self.dict_min_rwy_sep = dict_min_rwy_sep
        self.df_traffic_scenario = df_traffic_scenario


        self._trafficControl = trafficControl(self.dict_map_info,  self.control_config, self.dict_min_rwy_sep, random_seed)
        self.dict_rwy_free_time = ''
        self.dict_rwy_last_air = ''
        self.sumoFile = self.simFileLoc + '/' + sumoFilename + '.sumocfg'



        df_highest_OD = pd.read_csv(self.trafficFileLoc + 'update_highest_freq_OD.csv')

        self._findDepartInfo = findDepartInfo(self.control_config, random_seed, dict_rwy_id, df_highest_OD)

    def run_get_objective_iter(self, dict_state):

        dict_depart, df_depart = self._findDepartInfo.findDepartDictandDf(dict_state, self.df_traffic_scenario, self.dict_min_rwy_sep)
        # print(dict_depart)

        # make sure runway are already for landing at the begining
        self.dict_rwy_free_time = {k: 3600 for k, g in self.dict_map_info['dict_rwy_to_edge'].items()}
        self.dict_rwy_last_air = {k:'NA' for k, g in self.dict_map_info['dict_rwy_to_edge'].items()}

        traci.start([self.sumoBinary, "-c", self.sumoFile])

        number_of_conflicts, total_fuel_consumption, total_taxi_time_for_all_air, wait_time, df_speed = self.run_iter(df_depart, dict_depart)

        return number_of_conflicts, total_fuel_consumption, total_taxi_time_for_all_air, wait_time, df_speed

    def run_get_objective(self, dict_state):

        dict_depart, df_depart = self._findDepartInfo.findDepartDictandDf(dict_state, self.df_traffic_scenario, self.dict_min_rwy_sep)
        # print(dict_depart)

        # make sure runway are already for landing at the begining
        self.dict_rwy_free_time = {k: 3600 for k, g in self.dict_map_info['dict_rwy_to_edge'].items()}
        self.dict_rwy_last_air = {k:'NA' for k, g in self.dict_map_info['dict_rwy_to_edge'].items()}

        traci.start([self.sumoBinary, "-c", self.sumoFile])

        number_of_conflicts, total_fuel_consumption, total_taxi_time_for_all_air, wait_time = self.run(df_depart, dict_depart)

        return number_of_conflicts, total_fuel_consumption, total_taxi_time_for_all_air, wait_time


    def run_iter(self, df_depart, dict_depart):
        """execute the TraCI control loop"""
        # # 'E_10000' for mid air, 'E_10001' for hold gate air
        start_time = time.time()
        step = 0

        np_speed_loc = np.zeros([1, 10])  # first row is hallucinated

        self.dict_reset_route = {}
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            # print(step)
            # print(self.dict_rwy_free_time)

            self.dict_rwy_free_time = {k: v + 1 for k, v in self.dict_rwy_free_time.items()}

            # reset aircraft
            for veh_id, edge_list in self.dict_reset_route.items():
                traci.vehicle.setRoute(veh_id, edge_list)
            self.dict_reset_route = {}

            # get all current vehicle
            lst_air = traci.vehicle.getIDList()

            # # identify aircraft's roadID
            dict_air_with_road = {}

            for air_id in range(len(lst_air)):
                dict_air_with_road[lst_air[air_id]] = traci.vehicle.getRoadID(lst_air[air_id])
                # print("{} 's road list : {}".format(lst_air[air_id], traci.vehicle.getRoute(lst_air[air_id])))

            dict_control_air, self.dict_rwy_free_time, self.dict_rwy_last_air = self._trafficControl.controlAtStep(self.dict_rwy_free_time, self.dict_rwy_last_air, dict_air_with_road, step, df_depart, dict_depart)



            # print(dict_air_with_road)
            # print(dict_control_air)
            # print(step)

            for veh_id, veh_control in dict_control_air.items():
                # avoid automatically collision control in sumo simulation
                # traci.vehicle.setSpeedMode(veh_id,31)
                traci.vehicle.setSpeedMode(veh_id, 38)

                # set speed
                traci.vehicle.setSpeed(veh_id, veh_control['speed'])

                # set edge_list
                if veh_control['edge_list']=='none':
                    # print('found none : {} at step {}'.format(veh_id, step))
                    pass
                else:
                    traci.vehicle.moveToXY(vehID=veh_id, edgeID=veh_control['edge_list'][0], lane=0, keepRoute=2,
                                           x=traci.lane.getShape(veh_control['edge_list'][0] + '_0')[0][0],
                                           y=traci.lane.getShape(veh_control['edge_list'][0] + '_0')[0][1])
                    self.dict_reset_route[veh_id] = veh_control['edge_list']

                '''record each aircraft's speed and location'''
                np_new_speed_loc = self.get_speed_and_location(veh_id, veh_control, step=step)
                ## append each aircraft's speed and location
                if not np_new_speed_loc is None:
                    np_speed_loc = np.vstack([np_speed_loc, np_new_speed_loc])

            step = step + 1

            if step > 110000:
                break

        np_speed_loc = np.delete(np_speed_loc, 0, 0)  # delete first hallucinated row
        df = pd.DataFrame(data=np_speed_loc, columns=['time', 'id', 'speed', 'acceleration', 'edge_id', 'x', 'y', 'if_hold', 'operation', 'sum_free_rwy_time'])
        df['id'] = df['id'].astype(int).map(str)
        total_taxi_time_for_all_air = df['time'].max()

        lst_df = [g for _, g in df.groupby(['id'])]

        neg_airport_hourly_throughput = -1*(len(lst_df) / (total_taxi_time_for_all_air / 3600))

        number_of_conflicts = sum(list(map(self.count_conflict, lst_df)))

        lst_df = [(x, dict_depart) for x in lst_df]
        total_fuel_consumption = sum(list(map(self.count_fuel, lst_df)))

        # lst_fuel_df = [(len(x), dict_depart[str(int(x['id'].iloc[0]))]['fuel_rate']) for x in lst_df]
        # total_fuel_consumption = sum([x[0]*x[1] for x in lst_fuel_df])

        # total_delay_time = len(df[(df['if_hold']==1) | (df['if_hold']==2)])

        # total_congestion_time = len(df[df['if_hold'] == 5])

        wait_time = df['sum_free_rwy_time'].sum()

        traci.close()
        sys.stdout.flush()

        return number_of_conflicts, total_fuel_consumption, neg_airport_hourly_throughput, wait_time, df

    def run(self, df_depart, dict_depart):
        """execute the TraCI control loop"""
        # # 'E_10000' for mid air, 'E_10001' for hold gate air
        start_time = time.time()
        step = 0

        np_speed_loc = np.zeros([1, 10])  # first row is hallucinated

        self.dict_reset_route = {}
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            # print(step)
            # print(self.dict_rwy_free_time)

            self.dict_rwy_free_time = {k: v + 1 for k, v in self.dict_rwy_free_time.items()}

            # reset aircraft
            for veh_id, edge_list in self.dict_reset_route.items():
                traci.vehicle.setRoute(veh_id, edge_list)
            self.dict_reset_route = {}

            # get all current vehicle
            lst_air = traci.vehicle.getIDList()

            # # identify aircraft's roadID
            dict_air_with_road = {}

            for air_id in range(len(lst_air)):
                dict_air_with_road[lst_air[air_id]] = traci.vehicle.getRoadID(lst_air[air_id])
                # print("{} 's road list : {}".format(lst_air[air_id], traci.vehicle.getRoute(lst_air[air_id])))

            dict_control_air, self.dict_rwy_free_time, self.dict_rwy_last_air = self._trafficControl.controlAtStep(self.dict_rwy_free_time, self.dict_rwy_last_air, dict_air_with_road, step, df_depart, dict_depart)



            # print(dict_air_with_road)
            # print(dict_control_air)
            # print(step)

            for veh_id, veh_control in dict_control_air.items():
                # avoid automatically collision control in sumo simulation
                # traci.vehicle.setSpeedMode(veh_id,31)
                traci.vehicle.setSpeedMode(veh_id, 38)

                # set speed
                traci.vehicle.setSpeed(veh_id, veh_control['speed'])

                # set edge_list
                if veh_control['edge_list']=='none':
                    # print('found none : {} at step {}'.format(veh_id, step))
                    pass
                else:
                    traci.vehicle.moveToXY(vehID=veh_id, edgeID=veh_control['edge_list'][0], lane=0, keepRoute=2,
                                           x=traci.lane.getShape(veh_control['edge_list'][0] + '_0')[0][0],
                                           y=traci.lane.getShape(veh_control['edge_list'][0] + '_0')[0][1])
                    self.dict_reset_route[veh_id] = veh_control['edge_list']

                '''record each aircraft's speed and location'''
                np_new_speed_loc = self.get_speed_and_location(veh_id, veh_control, step=step)
                ## append each aircraft's speed and location
                if not np_new_speed_loc is None:
                    np_speed_loc = np.vstack([np_speed_loc, np_new_speed_loc])

            step = step + 1

            if step > 110000:
                break

        np_speed_loc = np.delete(np_speed_loc, 0, 0)  # delete first hallucinated row
        df = pd.DataFrame(data=np_speed_loc, columns=['time', 'id', 'speed', 'acceleration', 'edge_id', 'x', 'y', 'if_hold', 'operation', 'wait_time'])
        df['id'] = df['id'].astype(int).map(str)
        total_taxi_time_for_all_air = df['time'].max()

        lst_df = [g for _, g in df.groupby(['id'])]

        neg_airport_hourly_throughput = -1*(len(lst_df) / (total_taxi_time_for_all_air / 3600))

        number_of_conflicts = sum(list(map(self.count_conflict, lst_df)))

        lst_df = [(x, dict_depart) for x in lst_df]
        total_fuel_consumption = sum(list(map(self.count_fuel, lst_df)))

        # lst_fuel_df = [(len(x), dict_depart[str(int(x['id'].iloc[0]))]['fuel_rate']) for x in lst_df]
        # total_fuel_consumption = sum([x[0]*x[1] for x in lst_fuel_df])

        # total_delay_time = len(df[(df['if_hold']==1) | (df['if_hold']==2)])

        # total_congestion_time = len(df[df['if_hold'] == 5])

        wait_time = df['wait_time'].sum()

        traci.close()
        sys.stdout.flush()

        return number_of_conflicts, total_fuel_consumption, neg_airport_hourly_throughput, wait_time
    def count_conflict(self, df):
        if_hold_value = 5
        number_conflict = len([g for k, g in groupby(df['if_hold'].values) if k== if_hold_value])
        return number_conflict

    def count_fuel(self, mylist):
        df = mylist[0]
        dict_df = mylist[1][df['id'].iloc[0]]
        ### 0 for no hold, 1 for hold land, 2 for hold gate, 3 for hold inter for takoff, 4 for hold inter for cross, 5 for park###
        fuel_on_ground = len(df[~((df['if_hold'] == 1) | (df['if_hold'] == 2))]) * dict_df['fuel_rate_idle']
        fuel_in_air = len(df[df['if_hold'] == 1]) * dict_df['fuel_rate_approach']
        total_fuel = fuel_on_ground + fuel_in_air
        return total_fuel


    def get_speed_and_location(self, veh_id, veh_control, step=None):
        # store the speed information as time, id, speed, acceleration, edge_id, x, y, if_in_hold, operation
        road_id = veh_control['road_id']
        if (road_id == 'E_10000') and (veh_control['if_hold']==0):
            return None
        elif (road_id == 'E_10001') and (veh_control['if_hold']==0):
            return None
        else:
            np_air_speed = np.zeros([1, 10])
            np_air_speed[0, 0] = step
            np_air_speed[0, 1] = veh_id
            np_air_speed[0, 2] = veh_control['speed']
            np_air_speed[0, 3] = traci.vehicle.getAcceleration(veh_id)
            np_air_speed[0, 4] = int(road_id.split('_')[-1])
            air_position_x, air_position_y = traci.vehicle.getPosition(veh_id)
            np_air_speed[0, 5] = air_position_x
            np_air_speed[0, 6] = air_position_y
            np_air_speed[0, 7] = veh_control['if_hold']
            np_air_speed[0, 8] = self.dict_operation[traci.vehicle.getColor(veh_id)]
            np_air_speed[0, 9] = sum(self.dict_rwy_free_time.values())
            return np_air_speed


    def create_save_folder(self, control_config):
        dict_trafficVariationSettings = control_config['VariationSettings']
        lst_variationSettings = []
        for k, v in dict_trafficVariationSettings.items():
            lst_variationSettings.append(v)
        save_folder_name = '_'.join(lst_variationSettings)

        save_folder = 'result/' + save_folder_name
        save_folder_update = self.simFileLoc + save_folder
        createFolder(save_folder_update)
        return save_folder_update




if __name__ == '__main__':
    # variable settings:
        # speed @speed_variation
        # schedule @schedule_variation_type, @early_second, @late_second
        # route @route_variation_type
        # taxi way separation distance @col_dist
        # depreciated visualize collision distance @vis_dist

    control_config = configparser.ConfigParser()
    config = loadConfigForAirport('KLAX_test')
    trafficFileLoc = config['Path']['trafficInfoLoc']
    simFileLoc = config['Path']['simInfoLoc'] + 'highest_freq_traffic_demand/'
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

    # dict_state = dict(zip(lst_air_id, state))

    dict_state = {205: '07R/25L', 211: '06L/24R', 212: '06R/24L', 213: '07L/25R', 216: '06L/24R', 217: '06L/24R', 219: '07R/25L', 220: '06L/24R', 221: '07R/25L', 222: '07L/25R', 224: '06L/24R', 225: '07L/25R', 226: '06R/24L', 227: '06L/24R', 228: '07L/25R', 229: '06L/24R', 230: '06R/24L', 232: '07R/25L', 233: '07L/25R', 234: '06R/24L', 235: '07L/25R', 236: '07L/25R', 237: '07L/25R', 238: '07L/25R', 239: '06L/24R', 241: '07R/25L', 242: '06L/24R', 244: '07L/25R', 246: '07R/25L', 247: '06L/24R', 248: '06L/24R', 249: '07R/25L', 250: '06L/24R', 252: '07R/25L'}


    dict_min_rwy_sep = create_min_sep_time()

    traciSim = highestODSim(airport='KLAX_test', sumo_no_gui=True, option='highest_freq_traffic_demand',
                                 sim_file_id=1, sumoFilename=str(1), dict_rwy_id=dict_rwy_id,
                                 random_seed=1, dict_min_rwy_sep=dict_min_rwy_sep,
                                 df_traffic_scenario=df_traffic_scenario)

    number_of_conflicts, total_fuel_consumption, total_taxi_time_for_all_air = traciSim.run_get_objective(dict_state)
    print(number_of_conflicts, total_fuel_consumption, total_taxi_time_for_all_air)
    # now = datetime.datetime.now()
    # print("Current date and time : {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))
    # airport = 'KLAX_test'
    # _afterSimCal = afterSimCal(airport, simScenario='highest_freq_traffic_demand', col_dist=60,
    #                            sim_setting_file=1)
    #
    # now = datetime.datetime.now()
    # print("Current date and time : {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))

