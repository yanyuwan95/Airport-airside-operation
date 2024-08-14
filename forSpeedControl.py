import pandas as pd
import numpy as np
import random
# random.seed(seed=10)
from myFunction import *
import scipy

class speedControl(object):
    def __init__(self, dict_map_info, dict_option, random_seed):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        self.option = list(dict_option.keys())[0]
        self.option_value = dict_option[self.option]
        self.dict_rwy_to_edge = dict_map_info['dict_rwy_to_edge']



        if self.option == 'varOverAvgTaxiSpeed':
            self.avgSpeedDist = stats.lognorm(s=0.016523690117020498, loc=-145.12355377002302, scale=150.77634573054593)
            avgSpeed = self.avgSpeedDist.mean()
            stdSpeed = self.avgSpeedDist.std()
            self.lower = avgSpeed - self.option_value['variation'] * stdSpeed
            self.upper = avgSpeed + self.option_value['variation'] * stdSpeed
        elif self.option == 'randomSpeed':
            self.lower = self.option_value['lower']
            self.upper = self.option_value['upper']

    def calSpeed(self, road_id):
        if self.option == 'constantAvgSpeed':
            speed = self.constantAvgSpeed(self.option_value)
        elif self.option == 'randomSpeed':
            speed = self.randomSpeed(self.lower, self.upper)
        elif self.option == 'varOverAvgTaxiSpeed':
            speed = self.varOverAvgTaxiSpeed(self.avgSpeedDist, self.lower, self.upper)
        elif self.option == 'dependRoadId':
            speed = self.dependRoadId(road_id)
        return speed

    def constantAvgSpeed(self, constantValue):
        return constantValue

    def randomSpeed(self, lower, upper):
        speed = random.randint(lower, upper)
        return speed

    def dependRoadId(self, road_id):
        speed  = road_id/2
        return speed

    def varOverAvgTaxiSpeed(self, avgSpeedDist, lower, upper):
        # Gaussian: (5.675057116477112, 2.492128920644745)
        # LogNorm: (0.016523690117020498, -145.12355377002302, 150.77634573054593)
        while True:
            varSpeed = avgSpeedDist.rvs(size=1)
            if varSpeed < upper and varSpeed > lower:
                return varSpeed

    def return_speed(self, road_id):
        speed = self.calSpeed(road_id)
        return speed

if __name__ == '__main__':
    _speedControl = speedControl(dict_option={'constantAvgSpeed':5.67})
    speed = _speedControl.return_speed(road_id=1)
    print(speed)