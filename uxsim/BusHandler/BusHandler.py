
"""/*
Component added for TNDP using RL
*/"""
import numpy as np
import pandas as pd

class BusPassengerRequest:
    """
    BusHandler module for managing bus passenger requests and interactions.
    A class representing a bus passenger request (a person wanting to travel by bus)
    """
    def __init__(self, W, origin_stop, dest_stop, departure_time, name=None, attribute=None):
        """
        Create a bus passenger request.

        Parameters
        ----------
        W : World
            The world object.
        origin_stop : str | Node
            The origin bus stop node.
        dest_stop : str | Node
            The destination bus stop node.
        departure_time : float
            The time at which the passenger wants to depart.
        name : any, optional
            The name/identifier of the passenger request. Default is None.
        attribute : any, optional
            Additional attributes defined by users. Default is None.
        """
        
        self.W = W
        self.origin_stop = self.W.get_node(origin_stop)
        self.dest_stop = self.W.get_node(dest_stop)
        self.departure_time = departure_time
        self.name = name
        self.attribute = attribute

        # Trip timing attributes
        self.wait_start_time = None      # When passenger arrived at stop
        self.board_time = None           # When passenger boarded bus
        self.alight_time = None          # When passenger alighted bus
        self.total_travel_time = None    # Total door-to-door time
        self.wait_time = None            # Time spent waiting at origin stop
        self.in_vehicle_time = None      # Time spent on bus

        # Service attributes  
        self.bus = None                  # Which bus the passenger is currently on
        self.is_waiting = False          # Currently waiting at origin stop
        self.is_on_bus = False           # Currently traveling on bus
        self.trip_completed = False      # Has reached destination

    def __repr__(self):
        return f"BusPassengerRequest({self.name}: {self.origin_stop.name}→{self.dest_stop.name}, depart={self.departure_time})"


class BusHandler:
    """
    Base class for managing bus passenger requests and bus-passenger interactions.
    Handles the coordination between waiting passengers and bus services.
    """
    def __init__(self, W):
        """
        Create a bus handler.

        Parameters
        ----------
        W : World
            The world object.
        """
        self.W = W
        self.waiting_passengers = {}  # {stop_node: [BusPassengerRequest]}
        self.passenger_stats = []     # Completed passenger trip statistics
        self.pending_passengers = []  # Passengers waiting for their departure time
        self.bus_route_journeys = {}  # Track bus route progression: {bus_name: [stop_visits]}

    def handle_boarding_alighting(self, bus, current_stop):
        """
        Handle passenger boarding and alighting when a bus stops.

        Policy: A passenger should only board if their stop lies ahead along
        the current direction. Taxis/cars are point-to-point and don't have
        such direction constraints; buses do. We enforce "ahead-only" boarding
        for both circular and non-circular routes for simplicity.

        Parameters
        ----------
        bus : Vehicle
            The bus vehicle that is currently stopped.
        current_stop : Node
            The current bus stop node.
        """
        if bus.mode != "bus":
            return

        # Record capacity before any passenger activity
        capacity_before = len(bus.passengers)

        # Step 1: Alight passengers whose destination is current stop
        alighting_passengers = bus.alight_passengers(current_stop)
        num_alighted = len(alighting_passengers)
        
        # Step 2: Board waiting passengers (up to available capacity)
        boarded_passengers = []
        if current_stop in self.waiting_passengers:
            waiting_at_stop = self.waiting_passengers[current_stop]
            
            # Filter passengers who can be served by this bus and whose
            # destination lies ahead on the route from the current stop
            boardable_passengers = []
            for passenger in waiting_at_stop:
                if (
                    passenger.dest_stop in bus.route_stops
                    and self.can_reach_destination(bus, current_stop, passenger.dest_stop)
                ):
                    boardable_passengers.append(passenger)
            
            # Board passengers up to bus capacity
            boarded_passengers = bus.board_passengers(boardable_passengers)
            
            # Remove boarded passengers from waiting queue
            for passenger in boarded_passengers:
                self.waiting_passengers[current_stop].remove(passenger)
        
        num_boarded = len(boarded_passengers)
        capacity_after = len(bus.passengers)

        # Record bus route journey (track stop-to-stop progression)
        if bus.name not in self.bus_route_journeys:
            self.bus_route_journeys[bus.name] = []
        
        # Only record if this is a new stop or first visit to this stop
        journey = self.bus_route_journeys[bus.name]
        if not journey or journey[-1]['stop_name'] != current_stop.name:
            journey.append({
                'stop_name': current_stop.name,
                'arrival_time': self.W.TIME,
                'capacity_before': capacity_before,
                'passengers_alighted': num_alighted,
                'passengers_boarded': num_boarded,
                'capacity_after': capacity_after,
                'bus_capacity': getattr(bus, 'capacity')
            })
        
        # Step 3: Record statistics for completed trips
        for passenger in alighting_passengers:
            self.passenger_stats.append({
                'passenger_name': passenger.name,
                'origin_stop': passenger.origin_stop.name,
                'dest_stop': passenger.dest_stop.name,
                'departure_time': passenger.departure_time,
                'wait_time': passenger.wait_time,
                'in_vehicle_time': passenger.in_vehicle_time,
                'total_travel_time': passenger.total_travel_time,
                'bus_name': bus.name
            })

    def add_passenger_to_stop(self, passenger):
        """
        Add a passenger to the waiting queue at their origin stop.
        Called during simulation when passenger's departure time arrives.

        Parameters
        ----------
        passenger : BusPassengerRequest
            The passenger request to add to waiting queue.
        """
        origin_stop = passenger.origin_stop
        
        if origin_stop not in self.waiting_passengers:
            self.waiting_passengers[origin_stop] = []
            
        self.waiting_passengers[origin_stop].append(passenger)
        
        # Update passenger state (only when simulation is running)
        passenger.is_waiting = True
        if hasattr(self.W, 'TIME'):
            passenger.wait_start_time = self.W.TIME
        else:
            passenger.wait_start_time = passenger.departure_time

    def add_pending_passenger(self, passenger):
        """
        Add a passenger to the pending list during demand generation.
        They will be moved to waiting queues when their departure time arrives.

        Parameters
        ----------
        passenger : BusPassengerRequest
            The passenger request to add to pending list.
        """
        self.pending_passengers.append(passenger)

    def process_pending_passengers(self):
        """
        Check pending passengers and move them to waiting queues when their departure time arrives.
        Called during simulation timestep updates.
        """
        if not hasattr(self.W, 'TIME'):
            return
            
        # Find passengers whose departure time has arrived
        ready_passengers = [p for p in self.pending_passengers if p.departure_time <= self.W.TIME]
        
        # Move ready passengers to waiting queues
        for passenger in ready_passengers:
            self.add_passenger_to_stop(passenger)
            self.pending_passengers.remove(passenger)

    def can_reach_destination(self, bus, current_stop, dest_stop):
        """
        Direction-aware boarding check that respects road network connectivity.

        Returns True only if the destination stop is reachable by following the bus route
        and available road links. For back-and-forth routes, checks both forward and 
        backward directions. For circular routes, only allows forward direction.
        This is more sophisticated than simple index comparison as it validates actual
        road link existence between consecutive stops.

        Parameters
        ----------
        bus : Vehicle
            The bus vehicle being checked.
        current_stop : Node  
            The current bus stop node.
        dest_stop : Node
            The destination stop node.

        Returns
        -------
        bool
            True if destination is reachable via valid road connections, False otherwise.
        
        Notes
        -----
        - Intentionally stricter than allowing "ride-the-loop" on circular routes
        - Validates each link in the path exists in the road network
        - For back-and-forth routes: passenger can travel in either direction if links exist
        """
        try:
            current_idx = bus.route_stops.index(current_stop)
            dest_idx = bus.route_stops.index(dest_stop)
            
            if hasattr(bus, 'is_circular_route') and not bus.is_circular_route:
                # Back-and-forth route: check connectivity in either direction
                return (self._validate_path_links(bus, current_idx, dest_idx, forward=True) or
                        self._validate_path_links(bus, current_idx, dest_idx, forward=False))
            else:
                # Circular route: forward only, validate connectivity
                if dest_idx > current_idx:
                    return self._validate_path_links(bus, current_idx, dest_idx, forward=True)
                return False
        except ValueError:
            return False
    
    def _validate_path_links(self, bus, start_idx, dest_idx, forward):
        """
        Check if road links exist for path segment in specified direction
        """
        
        def has_road_link(node_a_name, node_b_name):
            node_a = self.W.get_node(node_a_name)
            node_b = self.W.get_node(node_b_name)
            return any(link.end_node == node_b for link in node_a.outlinks.values())
        
        if forward and dest_idx <= start_idx:
            return False
        if not forward and dest_idx >= start_idx:
            # For backward, check: current->end, then end->dest (reverse)
            stops = bus.route_stops
            # Check current to end (reverse direction)
            for i in range(start_idx, len(stops) - 1):
                if not has_road_link(stops[i + 1], stops[i]):
                    return False
            # Check end to destination (reverse direction)
            for i in range(len(stops) - 1, dest_idx, -1):
                if not has_road_link(stops[i], stops[i - 1]):
                    return False
            return True
        
        # Forward direction
        for i in range(start_idx, dest_idx):
            if not has_road_link(bus.route_stops[i], bus.route_stops[i + 1]):
                return False
        return True
    
    # TODO: Re-implement these functions.
    def compute_stats(self):
        """
        Compute bus system performance statistics.
        """
        pass

    def basic_to_pandas(self):
        """
        Convert basic bus system statistics to a pandas DataFrame.
        """
        pass
    
    def routes_to_pandas(self):
        """
        Convert route-specific statistics to a pandas DataFrame.

        
        """
        pass
    
    def passengers_to_pandas(self):
        """
        Convert passenger trip records to a pandas DataFrame.
        """
        pass
    
    def print_stats(self):
        """
        Print comprehensive bus system performance statistics.
        """
        pass

    def print_bus_activity_history(self):
        """
        Print bus activity history.
        """
        pass