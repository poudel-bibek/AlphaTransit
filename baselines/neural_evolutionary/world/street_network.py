import networkx as nx
import yaml

import config_utils

def build_network_graph(config_path):
    raise NotImplementedError()
    streets = nx.Graph()
    with open(config_path, 'r') as ff:
        config = yaml.load(ff)
    # read in the street network
    # build the street graph

    demand = nx.Graph()
    # TODO do we want different types of allowed OD data?  Nah, we can achieve
    # an OD-matrix-like effect with a properly scaled CSV.
    # read in the OD data
    # build the demand network


ALLOWED_HWY_TYPES = [
    "primary", 
    "primary_link", 
    "secondary", 
    "secondary_link", 
    "tertiary",
    "tertiary_link", 
    "trunk", 
    "trunk_link", 
    "motorway", 
    "motorway_link",
    "service"
]


# def build_graph_from


def build_graph_from_openstreetmap(network_path, show_modes=None):
    network_xml = config_utils.parse_xml(network_path)
    graph = nx.MultiDiGraph()

    for link_elem in network_xml.iter('{*}link'):
        id = link_elem.get('id')
        source = link_elem.get('from')
        dest = link_elem.get('to')
        freespeed = float(link_elem.get('freespeed'))
        length = float(link_elem.get('length'))
        drivetime = (length / freespeed)
        modes = link_elem.get('modes').split(',')
        attrs = {}
        for attr_elem in link_elem.iter('{*}attribute'):
            attrs[attr_elem.get('name')] = attr_elem.text
        attrs["drivetime"] = drivetime
        attrs["modes"] = modes
        attrs["length"] = length
        attrs["freespeed"] = freespeed
        if not show_modes or len(set(modes).union(set(show_modes))) > 0:
            hwytype = attrs.get("osm:way:highway", None)
            if hwytype in ALLOWED_HWY_TYPES:
                graph.add_edge(source, dest, **attrs)
            elif hwytype in ["residential", "unclassified"]:
                routes = attrs.get("osm:relation:route", "").split(",")
                if "bus" in routes:
                    graph.add_edge(source, dest, **attrs)
                elif hwytype == "unclassified" and 'bus' in modes:
                    graph.add_edge(source, dest, **attrs)

    for node_elem in network_xml.iter("{*}node"):
        xpos = float(node_elem.get("x"))
        ypos = float(node_elem.get("y"))
        id = node_elem.get("id")
        if id in graph.nodes:
            graph.nodes[id]['data'] = {"xpos": xpos, "ypos": ypos}

    return graph


def get_links(network_path):
    network_xml = config_utils.parse_xml(network_path)
    links = {}
    for link_elem in network_xml.iter('{*}link'):
        id = link_elem.get('id')
        source = link_elem.get('from')
        dest = link_elem.get('to')
        drivetime = (float(link_elem.get('length')) / 
            float(link_elem.get('freespeed')))
        modes = link_elem.get('modes').split(',')
        attrs = {}
        for attr_elem in link_elem.iter('{*}attribute'):
            attrs[attr_elem.get('name')] = attr_elem.text
        links[id] = {'source': source, 'dest': dest, 'drivetime': drivetime,
            'modes': modes, 'attributes': attrs}
    return links