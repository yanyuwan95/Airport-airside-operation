from SimSA_v3.myFunction import *
from SimSA_v3.TraciSim_highestODBased_get_df import highestODSim
import random
from itertools import product
import datetime
from multiprocessing import Pool
import itertools
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
SEED = 1
random.seed(SEED)
np.random.seed(SEED)

class optimizeRwySch(object):
    def __init__(self, airport, traffic_scenario, avg_run):
        config = loadConfigForAirport('KLAX_test')
        trafficFileLoc = config['Path']['trafficInfoLoc']
        self.avg_run = avg_run
        self.optimizeFileLoc = config['Path']['optimizeInfoLoc']
        self.simFileLoc = config['Path']['simInfoLoc']



        config_sim = ConfigParser()
        config_sim.read(self.simFileLoc + 'highest_freq_traffic_demand' + '/simulationSetting_1.ini')

        self.simulation_setting = str(config_sim['VariationSettings']['speedControl']) + '_' + str(config_sim['VariationSettings']['arrivalControl'] +'_'+str(config_sim['VariationSettings']['getPushControl']))
        self.simCacheLoc = self.optimizeFileLoc + 'sim_cache/'
        self.df_traffic_scenario = pd.read_csv(trafficFileLoc + 'scenario_'+str(traffic_scenario) + '.csv')
        self.dict_min_rwy_sep = self.create_min_sep_time()

        self.num_conflict_control = 5
        self.control_object_weight = np.array([0, 0.4, 0.6], dtype=np.float64)

        self.changed_state = True
        self.lst_last_dict_state = []

        self.n_core = 10
        self.objective_weight =  np.array([0.3, 0.3, 0.4],dtype=np.float64)  # number_of_conflicts, total_fuel_consumption, total_taxi_time_for_all_air
        self.best_dict_state = ''
        self.best_E = ''
        self.initial_E  = ''


        self.dict_rwy_id = {1:'07L/25R', 2: '07R/25L',3: '06R/24L', 4: '06L/24R'}
        self.n_rwys = len(self.dict_rwy_id)
        self.n_planes = len(self.df_traffic_scenario)

        self.df_traffic_scenario = self.df_traffic_scenario.rename(columns={'trajectory_id': 'veh_id'})
        self.df_traffic_scenario = self.add_initial_rwy(self.df_traffic_scenario)
        self.df_traffic_scenario = self.df_traffic_scenario.sort_values(by=['veh_id'])

        self.lst_air_id = self.df_traffic_scenario['veh_id'].values.tolist()

        self.traciSim = highestODSim(airport='KLAX_test', sumo_no_gui=True, option='highest_freq_traffic_demand',
                                     sim_file_id=1, sumoFilename=str(traffic_scenario),dict_rwy_id = self.dict_rwy_id,
                                     random_seed=1, dict_min_rwy_sep=self.dict_min_rwy_sep, df_traffic_scenario=self.df_traffic_scenario)

    def add_initial_rwy(self, df):
        dict_assign = {'06R/24L':[0,1,2,3,9,11,15,16], '07L/25R':[4,5,6,7,8,12,13,14,10],
                       '07R/25L':[4,5,6,7,8,12,13,14,10], '06L/24R':[0,1,2,3,9,11,15,16]}
        df['assign_initial_rwy'] = ''
        for i in range(len(df)):
            ramp_i = df['ramp_id'].iloc[i]
            operation_i = df['operation'].iloc[i]
            if operation_i=='A':
                if ramp_i in dict_assign['06L/24R']:
                    df['assign_initial_rwy'].iloc[i] = '06L/24R'
                else:
                    df['assign_initial_rwy'].iloc[i] = '07R/25L'
            else:
                if ramp_i in dict_assign['06R/24L']:
                    df['assign_initial_rwy'].iloc[i] = '06R/24L'
                else:
                    df['assign_initial_rwy'].iloc[i] = '07L/25R'

        return df



    def create_min_sep_time(self):
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

    # Define the objective function
    def objective_function(self, dict_state):
        total_obj = np.zeros(3, dtype=np.float64)
        for avg_i in range(self.avg_run):
            number_of_conflicts, total_fuel_consumption, neg_airport_hourly_throughput = self.traciSim.run_get_objective(
                dict_state)
            # Add current run's results to the total
            total_obj = total_obj + np.array(
                [number_of_conflicts, total_fuel_consumption, neg_airport_hourly_throughput],
                dtype=np.float64)
        avg_obj = total_obj / self.avg_run
        return avg_obj

    def objective_function_iter(self, dict_state, num_iteration):
        total_obj = np.zeros(3, dtype=np.float64)
        for avg_i in range(self.avg_run):
            number_of_conflicts, total_fuel_consumption, neg_airport_hourly_throughput, df_speed = self.traciSim.run_get_objective_iter(
                dict_state)
            df_speed.to_csv(self.simCacheLoc+'df_speed_' + str(num_iteration)+'_'+str(avg_i) + '.csv')
            # Add current run's results to the total
            total_obj = total_obj + np.array(
                [number_of_conflicts, total_fuel_consumption, neg_airport_hourly_throughput],
                dtype=np.float64)
        avg_obj = total_obj / self.avg_run
        return avg_obj

    # Define the neighbour function
    def neighbour(self, dict_state):
        continue_sample = True
        while continue_sample == True:
            new_state = list(dict_state.values())
            if len(self.lst_last_dict_state)==0:
                index = random.randint(0, int(self.n_planes-1))
                new_state[index] = self.dict_rwy_id[random.randint(1, self.n_rwys)]
                dict_state_new = dict(zip(dict_state.keys(), new_state))
                if dict_state_new in self.lst_last_dict_state:
                    continue_sample = True
                else:
                    continue_sample = False
                    return dict_state_new
            else:
                lst_index = random.sample(range(0,self.n_planes), min([int(len(self.lst_last_dict_state)/self.n_core), self.n_planes]))
                for index in lst_index:
                    new_state[index] = self.dict_rwy_id[random.randint(1, self.n_rwys)]
                dict_state_new = dict(zip(dict_state.keys(), new_state))
                if dict_state_new in self.lst_last_dict_state:
                    continue_sample = True
                else:
                    continue_sample = False
                    return dict_state_new
    def get_multi_neighbour(self, dict_state):
        lst_dict_state = []
        for i in range(self.n_core):
            lst_dict_state.append(self.neighbour(dict_state))
        return lst_dict_state


    # Define the acceptance probability function
    def acceptance_probability(self, dE, T, E):
        if all(dE < 0):
            return 1.0
        else:
            if E[0]< self.num_conflict_control:
                hard = np.prod(np.multiply(self.control_object_weight, np.exp(np.divide(-1 * dE, T))))
                soft = np.max(np.multiply(self.control_object_weight, np.exp(np.divide(-1 * dE, T))))
            else:
                hard = np.prod(np.multiply(self.objective_weight, np.exp(np.divide(-1 * dE, T))))
                soft = np.max(np.multiply(self.objective_weight, np.exp(np.divide(-1 * dE, T))))
            return 0.5 * hard + 0.5 * soft

    def find_optimal_objective_state(self, lst_new_energy, lst_dict_state_new, initial_E):

        lst_improve_energy = [np.divide(x - initial_E, np.multiply(np.array([1, 1, -1]), initial_E)) for x in lst_new_energy]

        lst_above_index = np.array(np.where(np.array(lst_new_energy)[:, 0] > self.num_conflict_control))

        lst_sum_energy =[(lambda x: np.sum(np.multiply(self.objective_weight, x)) if index in lst_above_index \
                        else np.sum(np.multiply(self.control_object_weight[1:], x[1:]))+(-1))(x) for index, x in enumerate(lst_improve_energy)]

        index_min = np.argmin(lst_sum_energy)

        dict_state_new = lst_dict_state_new[index_min]
        new_energy = lst_new_energy[index_min]

        # update best solution

        if all(lst_new_energy[index_min] < self.best_E):
            self.best_E = new_energy
            self.best_dict_state = dict_state_new
        elif lst_new_energy[index_min][0]< self.num_conflict_control and all(lst_new_energy[index_min][1:] < self.best_E[1:]):
            self.best_E = new_energy
            self.best_dict_state = dict_state_new

        return dict_state_new, new_energy

    def optimizeRwySche(self):

        max_iterations = 300  # maximum number of iterations
        initial_T = np.array([100, 100, 100],dtype=np.float64)  # initial temperature
        cooling_factor = 0.99  # cooling factor
        T_threshold = np.array([0.1, 0.1, 0.1],dtype=np.float64)  # temperature threshold

        np_result_store = np.zeros([1, int((self.n_core+2) * self.objective_weight.shape[0] + 1)])

        with open(self.optimizeFileLoc + self.simulation_setting+'.txt', 'w') as f:
            # Define the initial state of the system
            raw_state = [random.randint(1, self.n_rwys) for i in range(self.n_planes)]
            state = [self.dict_rwy_id[x] for x in raw_state]
            # state = self.df_traffic_scenario['assign_initial_rwy'].values.tolist()
            # state = self.df_traffic_scenario['rwy_id'].values.tolist()

            dict_state = dict(zip(self.lst_air_id, state))

            # Simulated annealing algorithm
            T = initial_T
            iteration = 0
            last_E = self.objective_function_iter(dict_state, 0)

            self.best_dict_state = dict_state
            self.best_E = last_E
            self.initial_E = last_E

            f.write("Initial solution: {}".format(dict_state))
            f.write('\n')
            f.write("Objective function value: {}".format(last_E))
            f.write('\n')
            print("Initial solution:", dict_state)
            print("Objective function value:", last_E)


            p = Pool(self.n_core)

            now = datetime.datetime.now()
            while all(T > T_threshold) and iteration < max_iterations:
                f.write('iteration is {}'.format(iteration))
                f.write('\n')
                print('iteration is {}'.format(iteration))
                # Generate a new state in the neighbourhood of the current state
                lst_dict_state_new = self.get_multi_neighbour(dict_state)

                # Evaluate the objective function for the new state
                E = last_E
                lst_new_energy = p.map(self.objective_function, lst_dict_state_new)

                f.write("lst_new_energy is: {}".format(lst_new_energy))
                f.write('\n')
                print('lst_new_energy is: {}'.format(lst_new_energy))

                dict_state_new, new_E = self.find_optimal_objective_state(lst_new_energy, lst_dict_state_new, self.initial_E)


                # Calculate the energy difference
                dE = new_E - E

                # Decide whether to accept the new state
                if self.acceptance_probability(dE, T, new_E) > random.random():
                    dict_state = dict_state_new
                    last_E = new_E
                    self.changed_state = True
                    self.lst_last_dict_state = []

                else:
                    self.changed_state = False
                    self.lst_last_dict_state.extend(lst_dict_state_new)

                f.write("current solution: {}".format(dict_state))
                f.write('\n')
                f.write("Objective function value: {}".format(new_E))
                f.write('\n')

                print("current solution:", dict_state)
                print("Objective function value:", last_E)

                # store result
                lst_new_energy.append(last_E)
                lst_new_energy.append(self.best_E)

                new_now = datetime.datetime.now()
                comp_seconds = (new_now -now).total_seconds()
                lst_new_energy.append(comp_seconds)

                np_store_i = np.hstack(lst_new_energy)




                np_result_store = np.vstack((np_result_store, np_store_i))

                # Reduce the temperature
                T *= cooling_factor

                # Increment the iteration counter
                iteration += 1
            p.close()
            # Print the best solution found
            f.write("Best solution: {}".format(self.best_dict_state))
            f.write('\n')
            f.write("Objective function value: {}".format(self.best_E))
            f.write('\n')

            print("Best solution:", self.best_dict_state)
            print("Objective function value:", self.best_E)

        best_E = self.objective_function_iter(self.best_dict_state, 300)


        lst_objective = ['number_of_conflicts', 'total_fuel_consumption', 'neg_airport_hourly_throughput']
        lst_result_column_name = []
        for i in range(0, self.n_core):
            lst_numbers = [i]
            lst_result_column_name_i = [f"{letter}_{number}" for letter, number in
                                        itertools.product(lst_objective, lst_numbers)]
            lst_result_column_name.extend(lst_result_column_name_i)


        lst_cur=['cur']
        lst_result_column_name_cur = [f"{letter}_{number}" for letter, number in itertools.product(lst_objective, lst_cur)]

        lst_opt = ['opt']
        lst_result_column_name_opt = [f"{letter}_{number}" for letter, number in
                                      itertools.product(lst_objective, lst_opt)]

        lst_result_column_name.extend(lst_result_column_name_cur)
        lst_result_column_name.extend(lst_result_column_name_opt)
        lst_result_column_name.append('comp_time')


        df_result = pd.DataFrame(data=np_result_store, columns=lst_result_column_name)

        df_result['diff_time'] = df_result['comp_time'].diff()

        df_result.to_csv(self.optimizeFileLoc+'result_' + self.simulation_setting+ '.csv', index=True, index_label='iteration')

if __name__ == '__main__':
    now = datetime.datetime.now()
    print("Current date and time : {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))
    _optimizeRwySch = optimizeRwySch(airport='KLAX_test', traffic_scenario='arrival', avg_run=10)
    _optimizeRwySch.optimizeRwySche()
    now = datetime.datetime.now()
    print("Current date and time : {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))

    # ['07R/25L', '06L/24R', '06R/24L', '07L/25R', '06L/24R', '06L/24R', '07R/25L', '06L/24R', '07R/25L', '07L/25R', '06L/24R', '07L/25R', '06R/24L', '06L/24R', '07L/25R', '06L/24R', '06R/24L', '07R/25L', '07L/25R', '06R/24L', '07L/25R', '07L/25R', '07L/25R', '07L/25R', '06L/24R', '07R/25L', '06L/24R', '07L/25R', '07R/25L', '06L/24R', '06L/24R', '07R/25L', '06L/24R', '07R/25L']