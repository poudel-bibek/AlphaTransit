
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
    def __init__(s, W, origin_stop, dest_stop, departure_time, name=None, attribute=None):
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
        
        s.W = W
        s.origin_stop = s.W.get_node(origin_stop)
        s.dest_stop = s.W.get_node(dest_stop)
        s.departure_time = departure_time
        s.name = name
        s.attribute = attribute

        # Trip timing attributes
        s.wait_start_time = None      # When passenger arrived at stop
        s.board_time = None           # When passenger boarded bus
        s.alight_time = None          # When passenger alighted bus
        s.total_travel_time = None    # Total door-to-door time
        s.wait_time = None            # Time spent waiting at origin stop
        s.in_vehicle_time = None      # Time spent on bus

        # Service attributes  
        s.bus = None                  # Which bus the passenger is currently on
        s.is_waiting = False          # Currently waiting at origin stop
        s.is_on_bus = False           # Currently traveling on bus
        s.trip_completed = False      # Has reached destination

    def __repr__(s):
        return f"BusPassengerRequest({s.name}: {s.origin_stop.name}→{s.dest_stop.name}, depart={s.departure_time})"


class BusHandler:
    """
    Base class for managing bus passenger requests and bus-passenger interactions.
    Handles the coordination between waiting passengers and bus services.
    """
    def __init__(s, W):
        """
        Create a bus handler.

        Parameters
        ----------
        W : World
            The world object.
        """
        s.W = W
        s.waiting_passengers = {}  # {stop_node: [BusPassengerRequest]} - passengers waiting at each stop
        s.passenger_stats = []     # List to store completed passenger trip statistics
        s.pending_passengers = []  # List of passengers waiting for their departure time

    def handle_boarding_alighting(s, bus, current_stop):
        """
        Handle passenger boarding and alighting when a bus stops.

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
        if current_stop in s.waiting_passengers:
            waiting_at_stop = s.waiting_passengers[current_stop]
            
            # Filter passengers who can be served by this bus (destination is on bus route)
            boardable_passengers = []
            for passenger in waiting_at_stop:
                if passenger.dest_stop in bus.route_stops:
                    boardable_passengers.append(passenger)
            
            # Board passengers up to bus capacity
            boarded_passengers = bus.board_passengers(boardable_passengers)
            
            # Remove boarded passengers from waiting queue
            for passenger in boarded_passengers:
                s.waiting_passengers[current_stop].remove(passenger)
        
        # Step 3: Record statistics for completed trips
        for passenger in alighting_passengers:
            s.passenger_stats.append({
                'passenger_name': passenger.name,
                'origin_stop': passenger.origin_stop.name,
                'dest_stop': passenger.dest_stop.name,
                'departure_time': passenger.departure_time,
                'wait_time': passenger.wait_time,
                'in_vehicle_time': passenger.in_vehicle_time,
                'total_travel_time': passenger.total_travel_time,
                'bus_name': bus.name
            })

    def add_passenger_to_stop(s, passenger):
        """
        Add a passenger to the waiting queue at their origin stop.
        Called during simulation when passenger's departure time arrives.

        Parameters
        ----------
        passenger : BusPassengerRequest
            The passenger request to add to waiting queue.
        """
        origin_stop = passenger.origin_stop
        
        if origin_stop not in s.waiting_passengers:
            s.waiting_passengers[origin_stop] = []
            
        s.waiting_passengers[origin_stop].append(passenger)
        
        # Update passenger state (only when simulation is running)
        passenger.is_waiting = True
        if hasattr(s.W, 'TIME'):
            passenger.wait_start_time = s.W.TIME
        else:
            passenger.wait_start_time = passenger.departure_time

    def add_pending_passenger(s, passenger):
        """
        Add a passenger to the pending list during demand generation.
        They will be moved to waiting queues when their departure time arrives.

        Parameters
        ----------
        passenger : BusPassengerRequest
            The passenger request to add to pending list.
        """
        s.pending_passengers.append(passenger)

    def process_pending_passengers(s):
        """
        Check pending passengers and move them to waiting queues when their departure time arrives.
        Called during simulation timestep updates.
        """
        if not hasattr(s.W, 'TIME'):
            return
            
        # Find passengers whose departure time has arrived
        ready_passengers = [p for p in s.pending_passengers if p.departure_time <= s.W.TIME]
        
        # Move ready passengers to waiting queues
        for passenger in ready_passengers:
            s.add_passenger_to_stop(passenger)
            s.pending_passengers.remove(passenger)
    
    def compute_stats(s):
        """
        Compute bus system performance statistics.
        
        This method calculates comprehensive statistics including:
        - Bus fleet metrics (distance, cycles, utilization)
        - Passenger service metrics (completed trips, waiting times)
        - Route-specific performance metrics
        """
        
        # Get all buses
        buses = [veh for veh in s.W.VEHICLES.values() if hasattr(veh, 'mode') and veh.mode == "bus"]
        
        # Bus fleet metrics
        s.n_buses = len(buses)
        s.total_bus_distance = sum(bus.distance_traveled for bus in buses)
        s.average_bus_distance = s.total_bus_distance / s.n_buses if s.n_buses > 0 else 0
        
        # Calculate route cycles and group buses by route
        s.buses_by_route = {}
        s.total_cycles = 0
        
        for bus in buses:
            # Estimate route length based on number of nodes and urban network characteristics
            # For Sioux Falls network: avg link length ≈ 800m (total 314km / 76 links ≈ 4km/link, but many parallel)
            if len(bus.route_path) > 1:
                avg_link_length = 800  # meters per link segment (reasonable for Sioux Falls)
                route_length = (len(bus.route_path) - 1) * avg_link_length
                
                # Add return segment for circular routes  
                if getattr(bus, 'is_circular_route', False):
                    route_length += avg_link_length
            else:
                route_length = 800  # Single node route (shouldn't happen, but safety)
            
            cycles = bus.distance_traveled / route_length if route_length > 0 else 0
            s.total_cycles += cycles
            
            # Group by route
            route_base = bus.name.split('_freq_')[0] if '_freq_' in bus.name else bus.name
            if route_base not in s.buses_by_route:
                s.buses_by_route[route_base] = {
                    'buses': [],
                    'total_distance': 0,
                    'total_passengers': 0,
                    'total_cycles': 0,
                    'frequency': getattr(bus, 'service_frequency', 1),
                    'route_length': route_length  # Store calculated route length (all buses in route should have same length)
                }
            else:
                # Use the route_length if current one is 0 but stored one isn't
                if route_length > 0:
                    s.buses_by_route[route_base]['route_length'] = route_length
            
            s.buses_by_route[route_base]['buses'].append(bus)
            s.buses_by_route[route_base]['total_distance'] += bus.distance_traveled
            s.buses_by_route[route_base]['total_passengers'] += len(bus.passengers)
            s.buses_by_route[route_base]['total_cycles'] += cycles
        
        # Waiting passengers at stops
        s.n_waiting_passengers = sum(len(passengers) for passengers in s.waiting_passengers.values())
        
        # Passenger service metrics (pending + waiting + completed = total)
        s.n_pending_passengers = len(s.pending_passengers)
        s.n_completed_passenger_trips = len(s.passenger_stats)
        s.n_total_passenger_requests = s.n_pending_passengers + s.n_waiting_passengers + s.n_completed_passenger_trips
        
        # Service rate
        s.passenger_service_rate = (s.n_completed_passenger_trips / s.n_total_passenger_requests * 100 
                                  if s.n_total_passenger_requests > 0 else 0)
        
        # Wait time statistics
        s.passenger_wait_times = [p['wait_time'] for p in s.passenger_stats]
        s.average_passenger_wait_time = (sum(s.passenger_wait_times) / len(s.passenger_wait_times) 
                                       if s.passenger_wait_times else 0)
        s.std_passenger_wait_time = np.std(s.passenger_wait_times) if len(s.passenger_wait_times) > 1 else 0
        
        # Bus utilization metrics
        s.current_bus_occupancy = sum(len(bus.passengers) for bus in buses)
        s.total_bus_capacity = sum(getattr(bus, 'capacity', 50) for bus in buses)
        s.fleet_utilization_rate = (s.current_bus_occupancy / s.total_bus_capacity * 100 
                                   if s.total_bus_capacity > 0 else 0)

    def basic_to_pandas(s):
        """
        Convert basic bus system statistics to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with basic bus system performance metrics
        """
        s.compute_stats()
        
        data = {
            "total_buses": [s.n_buses],
            "total_bus_distance_m": [s.total_bus_distance],
            "average_bus_distance_m": [s.average_bus_distance],
            "total_route_cycles": [s.total_cycles],
            "total_passenger_requests": [s.n_total_passenger_requests],
            "completed_passenger_trips": [s.n_completed_passenger_trips],
            "pending_passengers": [s.n_pending_passengers],
            "waiting_passengers": [s.n_waiting_passengers],
            "passenger_service_rate_percent": [s.passenger_service_rate],
            "average_passenger_wait_time_s": [s.average_passenger_wait_time],
            "current_bus_occupancy": [s.current_bus_occupancy],
            "total_bus_capacity": [s.total_bus_capacity],
            "fleet_utilization_percent": [s.fleet_utilization_rate]
        }
        
        return pd.DataFrame(data)
    
    def routes_to_pandas(s):
        """
        Convert route-specific statistics to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with performance metrics for each bus route
        """
        s.compute_stats()
        
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
        
        for route_name, route_data in s.buses_by_route.items():
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
    
    def passengers_to_pandas(s):
        """
        Convert passenger trip records to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with detailed passenger trip information
        """
        s.compute_stats()
        
        data = {
            "passenger_name": [p['passenger_name'] for p in s.passenger_stats],
            "origin_stop": [p['origin_stop'] for p in s.passenger_stats],
            "dest_stop": [p['dest_stop'] for p in s.passenger_stats],
            "departure_time": [p['departure_time'] for p in s.passenger_stats],
            "wait_time_s": [p['wait_time'] for p in s.passenger_stats],
            "in_vehicle_time_s": [p['in_vehicle_time'] for p in s.passenger_stats],
            "total_travel_time_s": [p['total_travel_time'] for p in s.passenger_stats],
            "bus_name": [p['bus_name'] for p in s.passenger_stats]
        }
        
        return pd.DataFrame(data)
    
    def print_stats(s):
        """
        Print comprehensive bus system performance statistics.
        """
        s.compute_stats()
        
        print("results for bus transportation:")
        
        print(f" total passenger requests: {s.n_total_passenger_requests}")
        print(f"   [Total people who wanted to travel by bus during simulation]\n")

        print(f" completed passenger trips: {s.n_completed_passenger_trips}")
        print(f"   [Passengers who successfully reached their destination]\n")

        print(f" passenger service rate: {s.passenger_service_rate:.1f}%")
        print(f"   [Percentage of demand successfully served]\n")

        print(f" passengers still pending: {s.n_pending_passengers}")
        print(f"   [People whose departure time hasn't arrived yet]\n")
        
        print(f" passengers waiting at stops: {s.n_waiting_passengers}")
        print(f"   [People currently waiting for buses at stops]\n")
        
        if s.passenger_wait_times:
            print(f" average passenger wait time: {s.average_passenger_wait_time:.1f} s")
            print(f"   [Average time passengers spent waiting at stops: {s.average_passenger_wait_time/60:.1f} minutes]")
            
            print(f" wait time std deviation: {s.std_passenger_wait_time:.1f} s")
            print(f"   [Variation in wait times - lower means more consistent service]\n")
        
        print(f" total buses in service: {s.n_buses}")
        print(f"   [Number of bus vehicles operating in the system]\n")
        
        print(f" total bus distance traveled: {s.total_bus_distance:.0f} m")
        print(f"   [Combined odometer reading of all buses: {s.total_bus_distance/1000:.1f} km total]\n")
        
        print(f" average distance per bus: {s.average_bus_distance:.0f} m")
        print(f"   [Typical distance per bus: {s.average_bus_distance/1000:.1f} km in {s.W.TMAX*s.W.DELTAT/3600:.1f} hours = {s.average_bus_distance/1000/(s.W.TMAX*s.W.DELTAT/3600):.1f} km/hr average speed]\n")
        
        print(f" estimated total route cycles: {s.total_cycles:.1f}")
        print(f"   [Number of complete route loops completed by all buses combined]\n")
        
        print(f" current fleet utilization: {s.fleet_utilization_rate:.1f}% ({s.current_bus_occupancy}/{s.total_bus_capacity})")
        print(f"   [Percentage of bus seats occupied at simulation end]\n")
        
        if s.buses_by_route:
            print(f" performance by route:")
            for route_name, route_data in s.buses_by_route.items():
                n_buses = len(route_data['buses'])
                avg_dist = route_data['total_distance'] / n_buses if n_buses > 0 else 0
                avg_passengers = route_data['total_passengers'] / n_buses if n_buses > 0 else 0
                route_length = route_data.get('route_length', 0)
                avg_cycles = route_data['total_cycles'] / n_buses if n_buses > 0 else 0
                print(f"  {route_name}: {n_buses} buses, {route_data['frequency']}/hr, {avg_dist:.0f}m avg, {avg_passengers:.1f} passengers avg")
                print(f"    [Route with {n_buses} buses running {route_data['frequency']} times/hour, each traveling {avg_dist/1000:.1f}km on average]")
                print(f"    route length: {route_length:.0f}m, avg cycles: {avg_cycles:.1f}")
                print(f"    [Each complete route loop is {route_length/1000:.1f}km, buses completed {avg_cycles:.1f} loops on average]\n")