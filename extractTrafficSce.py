import pandas as pd
import numpy as np
from yaml import safe_load
from myFunction import *


class mapFuelRate(object):
    # output: fuel rate for each aircraft "config['Path']['trafficInfoLoc'] + 'ICAO_fuel.csv'"
    def __init__(self, airport):
        self.fuelCount(airport)
    def fuelCount(self, airport):
        config = loadConfigForAirport(airport)
        air_type_File_Loc = config['Path']['trafficInfoLoc'] + 'aircraft'
        df_fuel_emission = pd.read_csv(config['Path']['trafficInfoLoc'] + 'fuel_emission.csv')

        df_air_type = pd.read_csv(config['Path']['trafficInfoLoc'] + 'FAA_aircraft_type.csv')

        df_fuel_emission = df_fuel_emission[['Fuel Flow Idle (kg/sec)', 'Engine Identification', 'Fuel Flow App (kg/sec)']].dropna()

        lst_air_file = get_file_name(air_type_File_Loc)
        df_engine = self.get_engine_dataframe(lst_air_file)

        df_engine['fuel_flow_idle_kg_sec'] = ''
        df_engine['fuel_flow_approach_kg_sec'] = ''
        df_engine['ICAO_type'] = ''
        for i in range(len(df_engine)):
            engine_i = df_engine['engine.default'].iloc[i]
            air_type_i = df_engine['air_type'].iloc[i]
            try:
                ICAO_type_i = df_air_type[df_air_type['ICAO Code'] == air_type_i]['Wake Category'].iloc[0]
                df_engine['ICAO_type'] = ICAO_type_i
            except:
                pass

            fuel_i = df_fuel_emission[df_fuel_emission['Engine Identification'].str.contains(engine_i)][
                'Fuel Flow Idle (kg/sec)'].iloc[0]
            df_engine['fuel_flow_idle_kg_sec'].iloc[i] = fuel_i

            fuel_i = df_fuel_emission[df_fuel_emission['Engine Identification'].str.contains(engine_i)][
                'Fuel Flow App (kg/sec)'].iloc[0]
            df_engine['fuel_flow_approach_kg_sec'].iloc[i] = fuel_i

        df_engine = df_engine.append(self.mannual_add_engine_air_ICAO(df_engine))

        df_engine.to_csv(config['Path']['trafficInfoLoc'] + 'ICAO_fuel.csv', index=False)

    def mannual_add_engine_air_ICAO(self, df):
        df_engine = pd.DataFrame(columns=df.columns)
        df_engine['engine.default'] = ['PW4056', 'PC6A']
        df_engine['air_type'] = ['B744', 'PC12']
        df_engine['fuel_flow_idle_kg_sec'] = [0.188, 0.0195]
        df_engine['fuel_flow_approach_kg_sec'] = [0.647, 0.07396]
        df_engine['ICAO_type'] = ['H', 'L']

        # https://mikeklochcfi.files.wordpress.com/2018/08/training-pt6a-60-series.pdf 90pph
        # https://en.wikipedia.org/wiki/Pilatus_PC-12
        # https: // apps.dtic.mil / sti / pdfs / ADA412301.pdf [Adopted]
        return df_engine
    def get_engine_dataframe(self, lst_air_file):
        df = pd.DataFrame()
        for file in lst_air_file:
            if '.yml' in file:
                with open(file, 'r') as f:
                    df_i = pd.json_normalize(safe_load(f))
                    df_i['air_type'] = file.split('.yml')[0].split('\\')[-1].upper()
                    df = df.append(df_i)
        df_engine = df[['engine.default', 'air_type']]
        return df_engine



class extractTrafficSce(object):
    def __init__(self, hour, star_minute, end_minute, input_save_file_name):
        self.input_save_file_name = input_save_file_name
        self.extract_schedule(hour, star_minute, end_minute)
    def extract_schedule(self, hour, star_minute, end_minute):
        config = loadConfigForAirport('KLAX_test')
        trafficFileLoc = config['Path']['trafficInfoLoc']
        traFileLoc = config['Path']['traInfoLoc']


        df_tra = pd.read_csv(traFileLoc + 'converted_mapping_result/converted_IFF_LAX+ASDEX_20170905_080008_86356_id_mapping_result.csv')

        df_tra = df_tra[(df_tra['start_time_in_hour']==hour) & (df_tra['start_time_in_minute']>=star_minute)& (df_tra['start_time_in_minute']<=end_minute)]

        df_tra_update = df_tra.drop_duplicates(subset=['trajectory_id'])
        df_tra_update['taxi_time'] = df_tra_update['end_time_stamp'] - df_tra_update['start_time_stamp']

        df_tra_update = df_tra_update[['trajectory_id', 'operation', 'ramp_id', 'rwy_id', 'taxi_time', 'reg_num', 'call_sign', 'start_time_in_minute']]

        df_tra_update['start_time_in_minute_update'] = df_tra_update['start_time_in_minute'] - df_tra_update['start_time_in_minute'].min()

        df_ori = pd.read_csv(traFileLoc + 'ori/IFF_LAX+ASDEX_20170905_080008_86356.csv', names=range(0, 35), low_memory=False)

        df_ori.columns = list(range(35))
        df_ori = df_ori.rename(columns={0: 'sign', 1: "time_stamp", 7: '2_call_sign',9: '3_latitude_2_type', 10: '3_longitude', 11: '3_height',
                                    12: '2_operation', 15: '2_reg_num', 16: '3_speed'})


        df_ICAO = pd.read_csv(trafficFileLoc + 'FAA_aircraft_type.csv')

        df_tra_update['air_type'] = ''
        df_tra_update['ICAO_type'] = ''
        for i in range(len(df_tra_update)):
            reg_num = df_tra_update['reg_num'].iloc[i]
            call_sign = df_tra_update['call_sign'].iloc[i]
            type = df_ori[(df_ori['sign']==2) & (df_ori['2_reg_num']==reg_num) & (df_ori['2_call_sign'] == call_sign)]['3_latitude_2_type'].iloc[0]
            df_tra_update['air_type'].iloc[i] = type

        for i in range(len(df_tra_update)):
            type = df_tra_update['air_type'].iloc[i]
            try:
                ICAO_type = df_ICAO[df_ICAO['ICAO Code']==type]['Wake Category'].iloc[0]
            except:
                ICAO_type = 'M'
            df_tra_update['ICAO_type'].iloc[i] = ICAO_type

        df_tra_update['ICAO_type'] = df_tra_update['ICAO_type'].fillna(value='M')

        df_fuel = pd.read_csv(trafficFileLoc + 'ICAO_fuel.csv')
        df_tra_update = self.add_fuel_count(df_tra_update, df_fuel)

        df_tra_update.to_csv(trafficFileLoc + self.input_save_file_name, index=False)

    def add_fuel_count(self, df, df_fuel):
        df['fuel_rate_idle'] = ''
        df['fuel_rate_approach'] = ''
        for i in range(len(df)):
            try:
                df['fuel_rate_idle'].iloc[i] = df_fuel[df_fuel['air_type'] == df['air_type'].iloc[i]]['fuel_flow_idle_kg_sec'].iloc[0]
                print('********found*************')
            except:
                ICAO_type = df['ICAO_type'].iloc[i]
                if ICAO_type == 'M':
                    df['fuel_rate_idle'].iloc[i] = df_fuel[df_fuel['air_type'] == 'A320']['fuel_flow_idle_kg_sec'].iloc[0]
                elif ICAO_type == 'H':
                    df['fuel_rate_idle'].iloc[i] = df_fuel[df_fuel['air_type'] == 'B744']['fuel_flow_idle_kg_sec'].iloc[0]
                elif ICAO_type == 'L':
                    df['fuel_rate_idle'].iloc[i] = df_fuel[df_fuel['air_type'] == 'PC12']['fuel_flow_idle_kg_sec'].iloc[0]


        for i in range(len(df)):
            try:
                df['fuel_rate_approach'].iloc[i] = df_fuel[df_fuel['air_type'] == df['air_type'].iloc[i]]['fuel_flow_approach_kg_sec'].iloc[0]
                print('********found*************')
            except:
                ICAO_type = df['ICAO_type'].iloc[i]
                if ICAO_type == 'M':
                    df['fuel_rate_approach'].iloc[i] = df_fuel[df_fuel['air_type'] == 'A320']['fuel_flow_approach_kg_sec'].iloc[0]
                elif ICAO_type == 'H':
                    df['fuel_rate_approach'].iloc[i] = df_fuel[df_fuel['air_type'] == 'B744']['fuel_flow_approach_kg_sec'].iloc[0]
                elif ICAO_type == 'L':
                    df['fuel_rate_approach'].iloc[i] = df_fuel[df_fuel['air_type'] == 'PC12']['fuel_flow_approach_kg_sec'].iloc[0]

        return df

if __name__ == '__main__':
    mapFuelRate('KLAX_test')
    df = extractTrafficSce(5, 0, 15, input_save_file_name='scenario_low.csv')


