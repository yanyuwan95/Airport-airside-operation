from configparser import ConfigParser


airport = 'KLAX_test'
#Get the configparser object
config_object = ConfigParser()
config_object.optionxform = lambda option: option

main_data_loc = "../../Data_v3/"

ori_data_loc = "../../oriTraj/"

config_object["Path"] = {"mainFileLoc": main_data_loc + airport,
                         "mapInfoLoc":main_data_loc + airport +"/mapInfo/",
                         "oriTrajLoc":ori_data_loc + airport,
                         "traInfoLoc":main_data_loc + airport +"/aircraftTraj/",
                         "simInfoLoc":main_data_loc + airport +"/simulation/",
                         "trafficInfoLoc": main_data_loc + airport +"/trafficAnalyze/",
                         "optimizeInfoLoc": main_data_loc + airport + "/optimization_v4/"}



#Write the above sections to config.ini file
with open(airport+'_beginLoadParameter.ini', 'w') as conf:
    config_object.write(conf)