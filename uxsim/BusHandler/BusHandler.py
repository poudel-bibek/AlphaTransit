
"""/*
Component added for TNDP using RL
Inclusion of transfer mechanism: 
    - Each passenger plans their trip: 
        - If a route is directly available, they will take it.
        - If a route is not directly available, they will get off at an intermediate stop and take a different route.
            - This will be done in a way that passengers take the shortest path (distance) to their destination.
            - The exact travel to the destination will not be known.
            - Transfer related stats will be recorded and published: 
                - Total number of transfers
                - The time passengers waited at the intermediate stop/s will also be recorded in their wait time. 
*/"""

import networkx as nx  # For shortest path in transit graph

class BusPassengerRequest:
    """
    Module for managing bus passenger requests and interactions
    Bus passenger request = a person wanting to travel by bus
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
            The time at which the passenger wants to depart from the origin stop.
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

        # Service attributes  
        self.bus = None                  # Which bus the passenger is currently on
        self.is_waiting = False          # Currently waiting at a stop (could be the origin stop or an intermediate transfer stop)
        self.is_on_bus = False           # Currently traveling on bus
        self.trip_completed = False      # Has reached destination
        
        self.itinerary = []              # The planned journey: [{'bus_route': str, 'board_stop': Node, 'alight_stop': Node}]
                                         # All passengers have an itinerary, even if they have no transfers.
        self.journey_log = []            # Log for all segments of the journey (in sequence): 
                                         # - Wait: {'type': 'wait', 'start': float, 'end': float or None, 'stop': str}
                                         # - Ride: {'type': 'ride', 'start': float, 'end': float or None, 'bus': str, 'board_stop': str, 'alight_stop': str or None}
        self.num_transfers = 0           # Total transfers in trip
        self.current_leg = 0             # Index of current itinerary leg

    def __repr__(self):
        return f"BusPassengerRequest({self.name}: {self.origin_stop.name}→{self.dest_stop.name}, depart={self.departure_time})"


class BusHandler:
    """
    Manage bus passenger requests and bus-passenger interactions.
    """
    def __init__(self, W):
        """
        Parameters
        ----------
        W : World
            The world object.
        """
        self.W = W
        self.waiting_passengers = {}  # {stop_node: [BusPassengerRequest]}
        self.passenger_stats = []     # Stats for passengers who have completed their trip
        self.pending_passengers = []  # Passengers waiting for their departure time
        self.unserviceable_passengers = []  # Passengers with no transit connectivity
        self.bus_route_journeys = {}  # Track bus route progression: {bus_name: [stop_visits]}

        # create a NetworkX graph of all bus stops/segments for per passener pathfinding (itinerary)
        self.transit_graph = None    # Built once with build_transit_graph

        # TODO: Instead of directly supplying the bus routes, they are populated in build_transit_graph by looking at buses.
        self.all_bus_routes = {}     # {bus_route: {'stops': [Node], 'is_circular': bool}} 

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

        # Capacity before any passenger activity
        capacity_before = len(bus.passengers)

        # Step 1: Alight passengers whose current leg alight or final dest is current stop
        alighting_passengers = []
        for p in bus.passengers[:]:  # Copy to avoid modification issues

            if current_stop == p.itinerary[p.current_leg]['alight_stop']:
                alighting_passengers.append(p)
                bus.passengers.remove(p)

                # Log alight (since passenger is alighting, the previous log must be 'ride' with end=None)
                p.journey_log[-1]['end'] = self.W.TIME
                p.journey_log[-1]['alight_stop'] = current_stop.name

                # Check if transfer or final
                if current_stop != p.dest_stop:
                    # Transfer
                    p.current_leg += 1
                    self.add_passenger_to_stop(current_stop, p)

                else:
                    # Final destination: calculate comprehensive multi-segment stats
                    # These values will be valid for multi-leg journeys.
                    total_travel = p.journey_log[-1]['end'] - p.journey_log[0]['start']
                    total_in_vehicle = sum(e['end'] - e['start'] for e in p.journey_log if e['type'] == 'ride')
                    total_wait = sum(e['end'] - e['start'] for e in p.journey_log if e['type'] == 'wait')
                    initial_wait = (p.journey_log[0]['end'] - p.journey_log[0]['start']) if p.journey_log[0]['type'] == 'wait' else 0
                    transfer_wait = total_wait - initial_wait

                    self.passenger_stats.append({
                        'passenger_name': p.name,
                        'origin_stop': p.origin_stop.name,
                        'dest_stop': p.dest_stop.name,
                        'departure_time': p.departure_time,
                        'total_travel_time': total_travel,
                        'total_in_vehicle_time': total_in_vehicle,
                        'total_wait_time': total_wait,
                        'transfer_wait_time': transfer_wait,
                        'num_transfers': p.num_transfers,
                        'journey_log': p.journey_log  # Full log for details
                    })
                    p.trip_completed = True

        num_alighted = len(alighting_passengers)

        # Step 2: Board waiting passengers (up to available capacity)
        boarded_passengers = []
        if current_stop in self.waiting_passengers:

            # Filter passengers who can be served by this bus
            boardable_passengers = []
            for passenger in self.waiting_passengers[current_stop]:
                leg = passenger.itinerary[passenger.current_leg]
                # names of buses are like bus_route_0, bus_route_0_freq_1, bus_route_0_freq_2, etc.
                if (bus.name == leg['bus_route'] or bus.name.startswith(leg['bus_route'] + '_')) and self.can_reach_destination(bus, current_stop, passenger):
                    boardable_passengers.append(passenger)

            # Board passengers up to bus capacity
            boarded_passengers = bus.board_passengers(boardable_passengers) # Not all of them will be boarded.
            
            # Remove boarded passengers from waiting queue and log boarding
            for passenger in boarded_passengers:
                self.waiting_passengers[current_stop].remove(passenger)
                # Close current wait
                passenger.journey_log[-1]['end'] = self.W.TIME
                # Add ride entry
                passenger.journey_log.append({'type': 'ride', 'start': self.W.TIME, 'end': None, 'bus': bus.name, 'board_stop': current_stop.name, 'alight_stop': None})

        num_boarded = len(boarded_passengers)
        capacity_after = len(bus.passengers)

        # Record bus route journey (track stop-to-stop progression)
        if bus.name not in self.bus_route_journeys:
            self.bus_route_journeys[bus.name] = []

        self.bus_route_journeys[bus.name].append({
            'stop_name': current_stop.name,
            'arrival_time': self.W.TIME,
            'capacity_before': capacity_before,
            'passengers_alighted': num_alighted,
            'passengers_boarded': num_boarded,
            'capacity_after': capacity_after,
            'bus_capacity': getattr(bus, 'capacity')
        })

    def add_passenger_to_stop(self, current_stop, passenger):
        """
        Add a passenger to the waiting queue.
        Called when:
            - passenger's departure time arrives at their origin stop.
            - passenger in transfer is alighted at an intermediate stop.

        Parameters
        ----------
        current_stop : Node
            The current bus stop node.
        passenger : BusPassengerRequest
            The passenger request to add to waiting queue.
        """
        
        if current_stop not in self.waiting_passengers:
            self.waiting_passengers[current_stop] = []
        
        self.waiting_passengers[current_stop].append(passenger) # Add passenger to waiting queue
        passenger.is_waiting = True # Update passenger state
        passenger.journey_log.append({'type': 'wait', 'start': self.W.TIME, 'end': None, 'stop': current_stop.name})

    def add_pending_passenger(self, passenger):
        """
        Add a passenger to the pending list during demand generation.
        They will be moved to waiting queues when their departure time arrives.
        Passengers with no transit connectivity are moved to unserviceable list.

        Parameters
        ----------
        passenger : BusPassengerRequest
            The passenger request to add to pending list.
        """
        self.assign_passenger_route(passenger)  # Assign route first
        
        if passenger.itinerary:
            # Passenger has valid transit route - add to pending
            self.pending_passengers.append(passenger)
        else:
            # Passenger has no transit connectivity - track as unserviceable
            self.unserviceable_passengers.append(passenger)

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
            self.add_passenger_to_stop(passenger.origin_stop, passenger)
            self.pending_passengers.remove(passenger)

    def has_road_link(self, node_a, node_b):
        """
        Helper: Check if there is a road link between two nodes.
        """
        return any(link.end_node == node_b for link in node_a.outlinks.values())

    def can_reach_destination(self, bus, current_stop, passenger):
        """
        Direction-aware boarding check that respects road network connectivity.

        Returns True only if the target stop is reachable by following the bus route
        and available road links. For back-and-forth routes, checks forward or backward 
        directions. For circular routes, only allows forward direction.

        Parameters
        ----------
        bus : Vehicle
            The bus vehicle being checked.
        current_stop : Node  
            The current bus stop node.
        passenger : BusPassengerRequest
            The passenger trying to board (provides itinerary for target).

        Returns
        -------
        bool
            True if target is reachable via valid road connections, False otherwise.

        Notes
        -----
        - Validates each link in the path exists in the road network.
        - For non-circular: Allows forward or backward if direct path exists.
        - For circular: Only forward, no wrapping around.
        """

        target = passenger.itinerary[passenger.current_leg]['alight_stop']
        if target == current_stop:
            return False  # Already at target, no need to board

        current_idx = bus.route_stops.index(current_stop)
        dest_idx = bus.route_stops.index(target)
        is_circular = getattr(bus, 'is_circular_route', False) # Check. Default is False.

        # Check forward path (for both circular and non-circular)
        if dest_idx > current_idx:
            for i in range(current_idx, dest_idx):
                if not self.has_road_link(bus.route_stops[i], bus.route_stops[i + 1]):
                    return False
            return True

        # If forward not found, check backward path (only for non-circular)
        elif dest_idx < current_idx and not is_circular:
            for i in range(current_idx, dest_idx, -1):
                if not self.has_road_link(bus.route_stops[i], bus.route_stops[i - 1]):
                    return False
            return True
        else:
            return False

    def build_transit_graph(self): 
        """
        A graph for passenger route assignment. 
        - Nodes = stops
        - Edges:
            - When two routes overlap at a stop, we add an edge between same set of nodes. 
                - This signifies that passengers can transfer between the two routes at this stop. 
                - i.e., the passenger can stay at the same node and change the route. 
                - Helps mark a transfer e.g., ['1'→'2' (Route0), '2'→'2' (transfer), '2'→'4' (Route1), '4'→'5' (Route1)]
            - Between different nodes: to allow travel between stops.
                - i.e., the passenger can travel from one stop to another stop on the same route. 
            - weights = distance in meters (zero for edge between same-stop).
        Call once after all buses are added to World.
        """

        self.transit_graph = nx.Graph()
        
        # The first bus on route will will be named bus_route_0 and other as bus_route_0_freq_1, bus_route_0_freq_2, etc.
        # The first bus on second route will will be named bus_route_1 and other as bus_route_1_freq_1, bus_route_1_freq_2, etc.
        for bus in [v for v in self.W.VEHICLES.values() if hasattr(v, 'mode') and v.mode == 'bus' and 'freq_' not in v.name]:  
            pattern = bus.name # e.g., 'bus_route_0'
            stops = bus.route_stops
            self.all_bus_routes[pattern] = {'stops': stops, 'is_circular': bus.is_circular_route}

            # Since we are progressively adding nodes to routes, the first time this is called, routes will have just two nodes.
            for i in range(len(stops) - 1):
                link = None  # Start with no link
                for out_link in stops[i].outlinks.values():  # Simple loop over outlinks
                    if out_link.end_node == stops[i+1]:
                        link = out_link  # Found it
                        break  # No need to check more
                distance = link.length if link else 0  # 0 if no link (or add min 100 if preferred)
                self.transit_graph.add_edge(stops[i].name, stops[i+1].name, weight=distance, bus_pattern=pattern)
                    
        # Add a self-loop transfer edge (zero weight) if a stop is present across two routes. 
        route_stop_count = {}
        for route in self.all_bus_routes.values():
            for stop in route['stops']:
                route_stop_count[stop.name] = route_stop_count.get(stop.name, 0) + 1

        for stop_name, count in route_stop_count.items():
            if count > 1:
                self.transit_graph.add_edge(stop_name, stop_name, weight=0, bus_pattern='transfer')  # Self-loop for transfers

    def assign_passenger_route(self, passenger):
        """
        Compute optimized shortest path itinerary for passenger using transit_graph.
        
        Strategy: Extend each route as far as possible, transfer at end of overlaps.
        This minimizes unnecessary transfers by staying on the current route through 
        overlapping segments instead of transferring at the first overlap node.
        
        Algorithm:
        1. Find shortest path using NetworkX (by distance weight)
        2. Convert path segments into journey legs
        3. For overlapping segments: stay on current route when possible
        4. Only transfer when current route cannot continue to destination
        
        Examples:
        ---------
        Basic journey:
            Path: ['1', '2', '3'] on route_0 
            → Itinerary: [{'board': '1', 'alight': '3', 'route': 'bus_route_0'}]
            → Transfers: 0
        
        Single transfer:
            Path: ['1', '2', '4', '5'] with route_0(1-2-3) and route_1(2-4-5)
            → Itinerary: [{'board': '1', 'alight': '2', 'route': 'bus_route_0'}, 
                          {'board': '2', 'alight': '5', 'route': 'bus_route_1'}]
            → Transfers: 1
        
        Optimized overlap handling:
            Path: ['1', '2', '3', '7'] with route_0(1-2-3), route_1(2-3-4), route_2(3-6-7)
            Behavior: 1→3 (route_0), 3→7 (route_2) = 1 transfer
        
        Unserviceable demand:
            Path: ['1', '8'] where node '8' not in transit_graph
            → Itinerary: [] (empty)
            → Passenger moved to unserviceable_passengers list
        
        Parameters
        ----------
        passenger : BusPassengerRequest
            The passenger requesting route assignment
            
        Updates
        -------
        passenger.itinerary : list
            Journey legs with board_stop, alight_stop, bus_route
        passenger.num_transfers : int  
            Number of transfers required (len(itinerary) - 1)
        """
        
        if not self.transit_graph:
            raise ValueError("Build transit_graph first.")
            
        orig = passenger.origin_stop.name
        dest = passenger.dest_stop.name
        try:
            path = nx.shortest_path(self.transit_graph, orig, dest, weight='weight') # based on distance
            
            # Convert path into journey legs using optimized overlap strategy
            itinerary = []
            current_route = None
            leg_start_stop = None
            
            for i in range(len(path) - 1):
                from_stop = path[i]
                to_stop = path[i + 1]
                edge_data = self.transit_graph.get_edge_data(from_stop, to_stop)
                
                # Skip transfer edges (passenger stays at same stop to change routes)
                if edge_data['bus_pattern'] == 'transfer':
                    continue
                    
                segment_route = edge_data['bus_pattern']
                
                # Start new leg if this is the first segment
                if current_route is None:
                    current_route = segment_route
                    leg_start_stop = from_stop
                    continue
                
                # If route changes, we need to decide: transfer now or continue?
                if segment_route != current_route:
                    # Check if current route can continue through this segment (overlap case)
                    current_route_can_continue = False
                    
                    # Check if current_route also serves this segment
                    all_edge_data = self.transit_graph.get_edge_data(from_stop, to_stop)
                    if isinstance(all_edge_data, dict) and 'bus_pattern' in all_edge_data:
                        # Single edge - check if current route has alternative path
                        for route_name, route_info in self.all_bus_routes.items():
                            if route_name == current_route:
                                route_stops = [stop.name for stop in route_info['stops']]
                                if from_stop in route_stops and to_stop in route_stops:
                                    # Check if they are consecutive in this route
                                    from_idx = route_stops.index(from_stop)
                                    to_idx = route_stops.index(to_stop)
                                    if abs(to_idx - from_idx) == 1:  # Consecutive stops
                                        current_route_can_continue = True
                                        break
                    
                    if current_route_can_continue:
                        # Stay on current route - this handles overlap case
                        continue  
                    else:
                        # Must transfer - finish current leg and start new one
                        itinerary.append({'board_stop': self.W.get_node(leg_start_stop), 'alight_stop': self.W.get_node(from_stop), 'bus_route': current_route})
                        current_route = segment_route
                        leg_start_stop = from_stop
            
            # Add the final leg
            if current_route is not None and leg_start_stop is not None:
                itinerary.append({'board_stop': self.W.get_node(leg_start_stop), 'alight_stop': self.W.get_node(path[-1]), 'bus_route': current_route })
            passenger.itinerary = itinerary
            passenger.num_transfers = max(0, len(itinerary) - 1)
            # print(f"\nItinerary: {[(leg['board_stop'].name, leg['alight_stop'].name, leg['bus_route']) for leg in itinerary]}")
            # print(f"Num transfers: {passenger.num_transfers}\n")
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            passenger.itinerary = []  # Unserviceable

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