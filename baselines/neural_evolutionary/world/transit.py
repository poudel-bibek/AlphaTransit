import copy

from lxml import etree
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from collections import defaultdict, OrderedDict
import torch

import config_utils


class RouteRep:
    """A class that holds a representation of a route with a cost, appropriate
      for our graph neural networks."""
    def __init__(self, route, cost, edge_index, edge_attr, route_idx=None):
        if type(route) is list:
            self.route = torch.tensor(route, device=edge_attr.device)
        else:
            self.route = route
        if type(cost) == torch.Tensor:
            self.cost = cost.item()
        else:
            self.cost = cost
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.route_idx = route_idx

    @property
    def norm_cost(self):
        return self.edge_attr[0, 1]

    @property
    def num_edges(self):
        return self.edge_attr.shape[0]

    @property
    def route_len(self):
        return len(self.route)

    @property
    def device(self):
        return self.edge_index.device

    @staticmethod
    def get_updated_costs_collection(route_reps, new_costs, min_cost=0,
                                     max_cost=273, normalize=True):
        if type(new_costs) is not torch.Tensor:
            new_costs = torch.tensor(new_costs, device=route_reps[0].device)
        route_reps = [copy.copy(rr) for rr in route_reps]
        if normalize:
            center = (max_cost + min_cost) / 2
            spread = max_cost - min_cost
            feat_costs = (new_costs - center) * 2 / spread
        else:
            feat_costs = new_costs
        for rr, cost, feat_cost in zip(route_reps, new_costs, feat_costs):
            rr.edge_attr = rr.edge_attr.clone()
            rr.edge_attr[:, 1] = feat_cost
            rr.cost = cost.item()
        return route_reps


class PtSystem:
    @classmethod
    def from_xml(cls, schedule_path, vehicles_path):
        # first read in the vehicle types
        vehicles_xml = config_utils.parse_xml(vehicles_path)
        types_dict = {}
        for vt_elem in vehicles_xml.iter('{*}vehicleType'):
            vehtype = PtVehicleType.from_matsim_xml(vt_elem)
            types_dict[vehtype.id] = vehtype

        # then the vehicles with reference to the types
        vehicles = [PtVehicle(ve.get('id'), types_dict[ve.get('type')])
                    for ve in vehicles_xml.iter('{*}vehicle')]
        vehicle_dict = {vv.id: vv for vv in vehicles}

        schedule_elem = config_utils.parse_xml(schedule_path).getroot()
        stop_facs = [PtStopFacility.from_matsim_xml(se)
                     for se in schedule_elem.iter('{*}stopFacility')]
        stop_facs = {sf.id: sf for sf in stop_facs}

        # then the routes with reference to the vehicles
        routes = []
        for line_elem in schedule_elem.iter('{*}transitLine'):
            line_id = line_elem.get('id')
            for route_elem in line_elem.iter('{*}transitRoute'):
                id = route_elem.get('id')
                mode = route_elem.find('{*}transportMode').text

                # build stop list
                stops = []
                for stop_elem in route_elem.iter('{*}stop'):
                    stop_fac = stop_facs[stop_elem.get('refId')]

                    arr_offset = stop_elem.get('arrivalOffset')
                    if arr_offset:
                        arr_offset = config_utils.str_time_to_float(arr_offset)

                    dep_offset = stop_elem.get('departureOffset')
                    if dep_offset:
                        dep_offset = config_utils.str_time_to_float(dep_offset)
                    stop = PtPlannedStop(stop_fac, arr_offset, dep_offset)
                    stops.append(stop)

                # build link list
                links = [le.get('refId') for le in route_elem.iter('{*}link')]

                # build deps list
                deps = []
                for dep_elem in route_elem.iter('{*}departure'):
                    dep_id = dep_elem.get('id')
                    dep_time = dep_elem.get('departureTime')
                    dep_time = config_utils.str_time_to_float(dep_time)
                    veh = vehicle_dict[dep_elem.get('vehicleRefId')]
                    deps.append(Departure(dep_id, dep_time, veh))

                # add the route to the list
                routes.append(PtRoute(id, line_id, mode, stops, links, deps))

        # finally, return the whole collection!
        return cls(routes)

    def __init__(self, pt_routes, pt_vehicle_types=None, pt_vehicles=None):
        # group the routes by their lines
        self._routes = {(rr.line_id, rr.route_id): rr for rr in pt_routes}
        self._routes_by_line = defaultdict(list)
        for route in self._routes.values():
            self._routes_by_line[route.line_id].append(route)

        # build the collection of vehicles
        self._vehicles = {}
        if pt_vehicles:
            for vehicle in pt_vehicles:
                self._vehicles[vehicle.id] = vehicle
        else:
            for route in self._routes.values():
                for departure in route.departures:
                    self._vehicles[departure.vehicle.id] = departure.vehicle

        # build the collection of vehicle types
        self._vehicle_types = {}
        if pt_vehicle_types:
            # The types were provided, so set them
            for vt in pt_vehicle_types:
                self._vehicle_types[vt.id] = vt
        else:
            # Infer the types from the vehicles
            for vehicle in self._vehicles.values():
                self._vehicle_types[vehicle.type.id] = vehicle.type

        # build the collection of unique stops
        self._stop_facilities = OrderedDict()
        for route in self._routes.values():
            for stop in route.stops:
                self._stop_facilities[stop.facility_id] = stop.facility

        # build a crow-flies distance matrix between stops
        num_stops = len(self._stop_facilities)
        dist_matrix = np.zeros((num_stops, num_stops))
        for ii, istop in enumerate(self._stop_facilities.values()):
            for jj, jstop in enumerate(self._stop_facilities.values()):
                diff = istop.position - jstop.position
                dist_matrix[ii, jj] = np.linalg.norm(diff)
        self.stop_dist_matrix = dist_matrix


    def get_route(self, line_id, route_id):
        return self._routes[(line_id, route_id)]

    def get_routes(self):
        return [vv for vv in self._routes.values()]

    def get_vehicle(self, vehicle_id):
        return self._vehicles[vehicle_id]

    def get_vehicles(self):
        return [vv for vv in self._vehicles.values()]

    def get_vehicle_type(self, vehicle_type_id):
        return self._vehicle_types[vehicle_type_id]

    def get_vehicle_types(self):
        return [vv for vv in self._vehicle_types.values()]

    def get_stop_facility(self, stop_fac_id):
        return self._stop_facilities[stop_fac_id]

    def get_stop_facilities(self):
        return [vv for vv in self._stop_facilities.values()]

    def get_stop_facility_ids(self):
        return [sf.id for sf in self.get_stop_facilities()]

    def write_as_xml(self, schedule_path, vehicles_path):
        sched_xml = config_utils.parse_xml(schedule_path)
        xmlroot = sched_xml.getroot()
        # remove existing lines
        [xmlroot.remove(le) for le in list(sched_xml.iter('{*}transitLine'))]

        # convert pt_route objects to xml
        for line_id, routes in self._routes_by_line.items():
            line_elem = etree.SubElement(xmlroot, 'transitLine')
            line_elem.set('id', line_id)
            for route in routes:
                if len(route.departures) == 0:
                    # routes with no departures crash matsim, so skip them
                    continue
                line_elem.append(route.to_xml())

        # write the schedule xml
        config_utils.write_xml(sched_xml, schedule_path)

        # remove the old vehicles
        vehicles_xml = config_utils.parse_xml(vehicles_path)
        xmlroot = vehicles_xml.getroot()
        [xmlroot.remove(vv) for vv in list(xmlroot.iter('{*}vehicle'))]

        # convert the new vehicles to xml
        for veh in self.get_vehicles():
            veh_elem = etree.SubElement(xmlroot, 'vehicle')
            veh_elem.set('id', veh.id)
            veh_elem.set('type', veh.type.id)

        # write the xml
        config_utils.write_xml(vehicles_xml, vehicles_path)


@dataclass
class PtVehicleType:
    """A struct for representing a type of public transit vehicle."""
    id: str
    mode: str
    description: str
    # number of seats
    seats: int
    # number of standing passengers the bus can hold
    standing_room: int
    length_m: float = 18
    # average power consumed by the vehicle while on a trip
    # default is estimate for a standard city bus
    avg_power_kW: float = 26.7

    @classmethod
    def from_matsim_xml(cls, vehicle_type_elem):
        id = vehicle_type_elem.get('id')
        cap_elem = vehicle_type_elem.find('{*}capacity')
        seats = cap_elem.find('{*}seats').get('persons')
        standing_room = cap_elem.find('{*}standingRoom').get('persons')
        length_m = vehicle_type_elem.find('{*}length').get('meter')
        desc_elem = vehicle_type_elem.find('{*}description')
        if desc_elem is not None:
            desc_text = desc_elem.text
        else:
            desc_text = ''
        desc_parts = desc_text.split('--')
        kwargs = {}
        if len(desc_parts) > 1:
            # try to get our custom values from the description string
            for part in desc_parts[1:]:
                key, _, value = part.partition('=')
                if key == 'avg_power_kW':
                    kwargs[key] = float(value)

        return cls(id, desc_text, seats, standing_room, length_m, **kwargs)

    def get_capacity(self):
        return self.seats + self.standing_room


@dataclass
class PtVehicle:
    id: str
    type: PtVehicleType


@dataclass
class PtStopFacility:
    id: str
    link: str
    _position: Tuple[float, float]
    # TODO replace with link object?  We don't have those yet

    def __post_init__(self):
        self._numpy_pos = np.array(self._position)

    @classmethod
    def from_matsim_xml(cls, xmlelem):
        xmldict = xmlelem.attrib
        pos = (float(xmldict['x']), float(xmldict['y']))
        return cls(xmldict['id'], xmldict['linkRefId'], pos)

    @property
    def position(self):
        return self._numpy_pos

    @position.setter
    def position(self, position):
        self._position = position
        self._numpy_pos = np.array(self._position)


@dataclass
class PtPlannedStop:
    """
    Represents a planned stop on a planned route (not an actual stop)
    """
    facility: PtStopFacility

    # one of these two must be set!
    _arrival_offset_s: float = None
    _departure_offset_s: float = None

    def __post_init__(self):
        if self._arrival_offset_s is None and self._departure_offset_s is None:
            raise ValueError('One of arrival_offset_s and departure_offset_s \
must be specified!')

    @property
    def facility_id(self):
        return self.facility.id

    @property
    def arrival_offset_s(self):
        if self._arrival_offset_s is None:
            return self._departure_offset_s
        else:
            return self._arrival_offset_s

    @property
    def departure_offset_s(self):
        if self._departure_offset_s is None:
            return self._arrival_offset_s
        else:
            return self._departure_offset_s


@dataclass
class Departure:
    id: str
    start_time_s: float
    vehicle: PtVehicle

@dataclass
class PtRoute:
    route_id: str
    line_id: str
    mode: str
    stops: List[PtPlannedStop]
    # TODO replace with link object.  We don't have those yet...
    links: List[str]
    departures: List[Departure] = None

    def __post_init__(self):
        self._stop_crowdist_matrix = None

    def to_xml(self):
        """Be aware that this just produces an element for this route, not for
        the line to which it belongs."""
        xml = etree.Element('transitRoute')
        # set id
        xml.set('id', self.route_id)
        # set transport mode
        mode_elem = etree.SubElement(xml, 'transportMode')
        mode_elem.text = self.mode

        # set stops
        stops_elem = etree.SubElement(xml, 'routeProfile')
        for stop in self.stops:
            stop_elem = etree.SubElement(stops_elem, 'stop')
            stop_elem.set('refId', stop.facility.id)
            if stop.arrival_offset_s is not None:
                time_str = config_utils.float_time_to_str(stop.arrival_offset_s)
                stop_elem.set('arrivalOffset', time_str)
            if stop.departure_offset_s is not None:
                time_str = config_utils.float_time_to_str(stop.departure_offset_s)
                stop_elem.set('departureOffset', time_str)

        # set links
        links_elem = etree.SubElement(xml, 'route')
        for link in self.links:
            link_elem = etree.SubElement(links_elem, 'link')
            link_elem.set('refId', link)

        # set departures
        deps_elem = etree.SubElement(xml, 'departures')
        for dep in self.departures:
            dep_elem = etree.SubElement(deps_elem, 'departure')
            dep_elem.set('id', dep.id)
            dep_elem.set('departureTime',
                         config_utils.float_time_to_str(dep.start_time_s))
            dep_elem.set('vehicleRefId', dep.vehicle.id)

        return xml

    def get_unique_id(self):
        # returns a tuple that is unique among routes on all lines
        return (self.line_id, self.route_id)

    def get_stop_crowdist_matrix(self):
        if self._stop_crowdist_matrix is not None:
            return self._stop_crowdist_matrix

        # compute crow-flies distances between stops on the route
        dists = np.zeros((len(self.stops), len(self.stops)))
        for ii, istop in enumerate(self.stops):
            for jj, jstop in enumerate(self.stops[ii + 1:], start=ii + 1):
                dists[ii, jj] = np.linalg.norm(istop.facility.position -
                                               jstop.facility.position)
                # make matrix symmetric
                dists[jj, ii] = dists[ii, jj]

        # store it to avoid recomputing later, as this is time-consuming
        self._stop_crowdist_matrix = dists
        return dists

    def get_estimated_travel_time_s(self):
        return self.stops[-1].arrival_offset_s


# TODO we may want code to generate a set of departures from a route and a set
# of vehicles.
