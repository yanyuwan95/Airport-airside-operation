import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from shapely import geometry
from shapely.geometry import Point, Polygon, LineString
import os
from multiprocessing import Pool
from itertools import repeat
from geopy import distance
from scipy.spatial import cKDTree
from datetime import date
import more_itertools as mit
from itertools import groupby
from operator import itemgetter
from pyproj import Proj, transform
from rdp import rdp
import time
import sys

class Graph_map_infor(object):

    # In networkx nodes are int;
    # In LAP_map.csv nodes are int.
    def __init__(self, G):
        self.G = G
        self.max_node_osmid = max(list(G.nodes))

    def convert_node_attr_types(self, node_dtypes=None):
        """
        Convert graph nodes' attributes' types from string to numeric.
        Parameters
        ----------
        G : networkx.MultiDiGraph
            input graph
        node_type : type
            convert node ID (osmid) to this type
        node_dtypes : dict of attribute name -> data type
            identifies additional is a numpy.dtype or Python type
            to cast one or more additional node attributes
            defaults to {"elevation":float, "elevation_res":float,
            "lat":float, "lon":float, "x":float, "y":float} if None
        Returns
        -------
        G : networkx.MultiDiGraph
        """
        if node_dtypes is None:
            node_dtypes = {"highway": str, "ref": str, "osmid": str,
                           "x": float, "y": float}
        for _, data in self.G.nodes(data=True):

            # convert numeric node attributes from string to float
            for attr in node_dtypes:
                if attr in data:
                    dtype = node_dtypes[attr]
                    data[attr] = dtype(data[attr])
        return self.G

    def convert_edge_attr_types(self, edge_dtypes=None):
        """
        Convert graph edges' attributes' types from string to numeric.
        Parameters
        ----------
        G : networkx.MultiDiGraph
            input graph
        node_type : type
            convert osmid to this type
        edge_dtypes : dict of attribute name -> data type
            identifies additional is a numpy.dtype or Python
            type to cast one or more additional edge attributes.
        Returns
        -------
        G : networkx.MultiDiGraph
        """
        if edge_dtypes is None:
            edge_dtypes = {"bridge": str, "geometry": LineString, "name": str,
                           "length": float, "oneway": str, "width": str,
                           "ref": str, "osmid": str}
        # convert numeric, bool, and list edge attributes from string
        # to correct data types

        # convert to specfied dtype any possible OSMnx-added edge attributes, which may
        # have multiple values if graph was simplified after they were added
        for _, _, data in self.G.edges(data=True, keys=False):

            # convert to specfied dtype any possible OSMnx-added edge attributes, which may
            # have multiple values if graph was simplified after they were added
            for attr in edge_dtypes:
                if attr in data:
                    dtype = edge_dtypes[attr]
                    data[attr] = dtype(data[attr])
        return self.G