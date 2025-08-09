
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

    def handle_boarding_alighting(self, bus, current_stop):
        """
        Handle passenger boarding and alighting when a bus stops.

        Policy: A passenger should only board if their stop lies ahead along
        the current direction. Taxis/cars are point-to-point and don’t have
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

        # Step 1: Alight passengers whose destination is current stop
        alighting_passengers = bus.alight_passengers(current_stop)
        
        # Step 2: Board waiting passengers (up to available capacity)
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
        Direction-aware boarding check.

        Returns True only if the destination stop appears after the current
        stop in the bus.route_stops order. This enforces that a passenger only
        boards when their destination lies ahead along the current direction
        of travel. For simplicity, the same policy is used for circular and
        non-circular routes.

        Notes
        -----
        - This is intentionally stricter than allowing "ride-the-loop" on
          circular routes, to avoid capacity pollution and unrealistic trips.
        - Taxis/cars are point-to-point and do not use this constraint.
        """
        try:
            current_idx = bus.route_stops.index(current_stop)
            dest_idx = bus.route_stops.index(dest_stop)
            return dest_idx > current_idx
        except ValueError:
            return False
    
    def compute_stats(self):
        """
        Compute bus system performance statistics.
        
        This method calculates comprehensive statistics including:
        - Bus fleet metrics (distance, cycles, utilization)
        - Passenger service metrics (completed trips, waiting times)
        - Route-specific performance metrics
        """
        
        # Get all buses
        buses = [veh for veh in self.W.VEHICLES.values() if hasattr(veh, 'mode') and veh.mode == "bus"]
        
        # Bus fleet metrics
        self.n_buses = len(buses)
        self.total_bus_distance = sum(bus.distance_traveled for bus in buses)
        self.average_bus_distance = self.total_bus_distance / self.n_buses if self.n_buses > 0 else 0

        # Calculate exact route length from actual links; group buses by route
        # Assumption: set_bus_route ensures a connected route; missing links are treated as errors.
        self.buses_by_route = {}
        self.total_cycles = 0

        for bus in buses:
            if len(getattr(bus, 'route_path', [])) <= 1:
                raise ValueError(f"Bus {bus.name} has an invalid route_path length: {len(getattr(bus, 'route_path', []))}")

            # Build consecutive node pairs from the route path
            node_pairs = []
            rp = bus.route_path
            for i in range(len(rp) - 1):
                node_pairs.append((rp[i], rp[i + 1]))
            # For circular routes, include wrap-around link
            if getattr(bus, 'is_circular_route', False):
                node_pairs.append((rp[-1], rp[0]))

            # Sum exact link lengths along the route
            route_length = 0.0
            for a, b in node_pairs:
                link = None
                # Use outlinks from node 'a' to find the directed link to 'b'
                for l in a.outlinks.values():
                    if l.end_node is b:
                        link = l
                        break
                if link is None:
                    raise ValueError(f"Route for bus {bus.name} is not connected between {a.name} -> {b.name}")
                route_length += float(link.length)

            if route_length <= 0:
                raise ValueError(f"Computed non-positive route length for bus {bus.name}")

            cycles = bus.distance_traveled / route_length
            self.total_cycles += cycles

            # Group by route base name
            route_base = bus.name.split('_freq_')[0] if '_freq_' in bus.name else bus.name
            if route_base not in self.buses_by_route:
                self.buses_by_route[route_base] = {
                    'buses': [],
                    'total_distance': 0.0,
                    'total_passengers': 0,
                    'total_cycles': 0.0,
                    'frequency': getattr(bus, 'service_frequency', 1),
                    'route_length': route_length,
                }
            else:
                # Overwrite with exact length (all buses on same named route should share the same path)
                self.buses_by_route[route_base]['route_length'] = route_length

            self.buses_by_route[route_base]['buses'].append(bus)
            self.buses_by_route[route_base]['total_distance'] += bus.distance_traveled
            self.buses_by_route[route_base]['total_passengers'] += len(bus.passengers)
            self.buses_by_route[route_base]['total_cycles'] += cycles
        
        # Waiting passengers at stops
        self.n_waiting_passengers = sum(len(passengers) for passengers in self.waiting_passengers.values())
        
        # Passenger service metrics (pending + waiting + completed = total)
        self.n_pending_passengers = len(self.pending_passengers)
        self.n_completed_passenger_trips = len(self.passenger_stats)
        self.n_total_passenger_requests = self.n_pending_passengers + self.n_waiting_passengers + self.n_completed_passenger_trips
        
        # Service rate
        self.passenger_service_rate = (self.n_completed_passenger_trips / self.n_total_passenger_requests * 100 
                                  if self.n_total_passenger_requests > 0 else 0)
        
        # Wait time statistics
        self.passenger_wait_times = [p['wait_time'] for p in self.passenger_stats]
        self.average_passenger_wait_time = (sum(self.passenger_wait_times) / len(self.passenger_wait_times) 
                                       if self.passenger_wait_times else 0)
        self.std_passenger_wait_time = np.std(self.passenger_wait_times) if len(self.passenger_wait_times) > 1 else 0
        
        # Bus utilization metrics
        self.current_bus_occupancy = sum(len(bus.passengers) for bus in buses)
        self.total_bus_capacity = sum(getattr(bus, 'capacity', 50) for bus in buses)
        self.fleet_utilization_rate = (self.current_bus_occupancy / self.total_bus_capacity * 100 
                                   if self.total_bus_capacity > 0 else 0)

    def basic_to_pandas(self):
        """
        Convert basic bus system statistics to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with basic bus system performance metrics
        """
        self.compute_stats()
        
        data = {
            "total_buses": [self.n_buses],
            "total_bus_distance_m": [self.total_bus_distance],
            "average_bus_distance_m": [self.average_bus_distance],
            "total_route_cycles": [self.total_cycles],
            "total_passenger_requests": [self.n_total_passenger_requests],
            "completed_passenger_trips": [self.n_completed_passenger_trips],
            "pending_passengers": [self.n_pending_passengers],
            "waiting_passengers": [self.n_waiting_passengers],
            "passenger_service_rate_percent": [self.passenger_service_rate],
            "average_passenger_wait_time_s": [self.average_passenger_wait_time],
            "current_bus_occupancy": [self.current_bus_occupancy],
            "total_bus_capacity": [self.total_bus_capacity],
            "fleet_utilization_percent": [self.fleet_utilization_rate]
        }
        
        return pd.DataFrame(data)
    
    def routes_to_pandas(self):
        """
        Convert route-specific statistics to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with performance metrics for each bus route
        """
        self.compute_stats()
        
        data = {
            "route_name": [],
            "n_buses": [],
            "service_frequency_per_hour": [],
            "total_distance_m": [],
            "average_distance_per_bus_m": [],
            "total_cycles": [],
            "current_passengers": [],
            "average_passengers_per_bus": []
        }
        
        for route_name, route_data in self.buses_by_route.items():
            n_buses = len(route_data['buses'])
            data["route_name"].append(route_name)
            data["n_buses"].append(n_buses)
            data["service_frequency_per_hour"].append(route_data['frequency'])
            data["total_distance_m"].append(route_data['total_distance'])
            data["average_distance_per_bus_m"].append(route_data['total_distance'] / n_buses if n_buses > 0 else 0)
            data["total_cycles"].append(route_data['total_cycles'])
            data["current_passengers"].append(route_data['total_passengers'])
            data["average_passengers_per_bus"].append(route_data['total_passengers'] / n_buses if n_buses > 0 else 0)
        
        return pd.DataFrame(data)
    
    def passengers_to_pandas(self):
        """
        Convert passenger trip records to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with detailed passenger trip information
        """
        self.compute_stats()
        
        data = {
            "passenger_name": [p['passenger_name'] for p in self.passenger_stats],
            "origin_stop": [p['origin_stop'] for p in self.passenger_stats],
            "dest_stop": [p['dest_stop'] for p in self.passenger_stats],
            "departure_time": [p['departure_time'] for p in self.passenger_stats],
            "wait_time_s": [p['wait_time'] for p in self.passenger_stats],
            "in_vehicle_time_s": [p['in_vehicle_time'] for p in self.passenger_stats],
            "total_travel_time_s": [p['total_travel_time'] for p in self.passenger_stats],
            "bus_name": [p['bus_name'] for p in self.passenger_stats]
        }
        
        return pd.DataFrame(data)
    
    def print_stats(self):
        """
        Print comprehensive bus system performance statistics.
        """
        self.compute_stats()
        
        print("\n\nResults for bus transportation:")
        
        print(f" total passenger requests: {self.n_total_passenger_requests}")
        print(f"   [Total people who wanted to travel by bus during simulation]\n")

        print(f" completed passenger trips: {self.n_completed_passenger_trips}")
        print(f"   [Passengers who successfully reached their destination]\n")

        print(f" passenger service rate: {self.passenger_service_rate:.1f}%")
        print(f"   [Percentage of demand successfully served]\n")

        print(f" passengers still pending: {self.n_pending_passengers}")
        print(f"   [People whose departure time hasn't arrived yet]\n")
        
        print(f" passengers waiting at stops: {self.n_waiting_passengers}")
        print(f"   [People currently waiting for buses at stops]\n")
        
        if self.passenger_wait_times:
            print(f" average passenger wait time: {self.average_passenger_wait_time:.1f} s")
            print(f"   [Average time passengers spent waiting at stops: {self.average_passenger_wait_time/60:.1f} minutes]")
            
            print(f" wait time std deviation: {self.std_passenger_wait_time:.1f} s")
            print(f"   [Variation in wait times - lower means more consistent service]\n")
        
        print(f" total buses in service: {self.n_buses}")
        print(f"   [Number of bus vehicles operating in the system]\n")
        
        print(f" total bus distance traveled: {self.total_bus_distance:.0f} m")
        print(f"   [Combined odometer reading of all buses: {self.total_bus_distance/1000:.1f} km total]\n")
        
        print(f" average distance per bus: {self.average_bus_distance:.0f} m")
        print(f"   [Typical distance per bus: {self.average_bus_distance/1000:.1f} km in {self.W.TMAX*self.W.DELTAT/3600:.1f} hours = {self.average_bus_distance/1000/(self.W.TMAX*self.W.DELTAT/3600):.1f} km/hr average speed]\n")
        
        print(f" estimated total route cycles: {self.total_cycles:.1f}")
        print(f"   [Number of complete route loops completed by all buses combined]\n")
        
        print(f" current fleet utilization: {self.fleet_utilization_rate:.1f}% ({self.current_bus_occupancy}/{self.total_bus_capacity})")
        print(f"   [Percentage of bus seats occupied at simulation end]\n")
        
        if self.buses_by_route:
            print(f" performance by route:")
            for route_name, route_data in self.buses_by_route.items():
                n_buses = len(route_data['buses'])
                avg_dist = route_data['total_distance'] / n_buses if n_buses > 0 else 0
                avg_passengers = route_data['total_passengers'] / n_buses if n_buses > 0 else 0
                route_length = route_data.get('route_length', 0)
                avg_cycles = route_data['total_cycles'] / n_buses if n_buses > 0 else 0
                print(f"  {route_name}: {n_buses} buses (from service_frequency={route_data['frequency']}/hr), {avg_dist:.0f}m avg, {avg_passengers:.1f} current passengers/bus (end)")
                print(f"    [Bus count reflects service_frequency and simulation horizon.")
                print(f"    route length: {route_length:.0f}m, avg cycles: {avg_cycles:.1f}")
                print(f"    [Each complete route loop is {route_length/1000:.1f}km, buses completed {avg_cycles:.1f} loops on average]\n")