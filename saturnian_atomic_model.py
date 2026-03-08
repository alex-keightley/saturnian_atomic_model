# Library Imports
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
k = 8.9875517923e9 # Coulomb's constant; N*m^2/C^2
q = 1.602*10**-19 # Magnitude of base electron charge [C]
m = 9.11*10**-31 # Base electron mass [kg]

# Bohr (or Nagaoka-Keightley) radius
r_bohr = 5.29*10**-11 # [m]

# Instantiation of variables used to determine solve_ivp max_step, a baseline value for orbit period T to be used when determining how long
# to run each simulation.
r0 = r_bohr 
v0 = np.sqrt(k * q**2 / (m * r0))

# Calculate period of orbit on which to base our simulation run time and solve_ivp max_step iteration
T = 2*np.pi*r0/v0

# Set max step such there are at least 1000 steps per orbit (based on Bohr radius and velocity for circular orbit)
max_step = T*10**-3

# Set tolerances for solve_ivp
atol = 1e-8
rtol = 1e-8

# Set the location of the fixed charge to <0,0>
fixed_charges = [0, 0]

# Instantiate a global colours array in order to assign colours to orbiting electrons for dynamically sized state vectors
colours = ['g', 'r', 'c', 'm', 'y', 'k']

# Instantiate a global linestyle array in order to assign linestyles to orbiting electrons for dynamically sized state vectors
line_styles = ['-', '--', '-.', ':']

# The function create_state0 is a function that takes in input arguments and returns a state0 vector. The create_state0 function
# takes in initial arguments
#     n - number of electrons for which to build an initial state vector, where the length of state0 will be 4*n
#     r_bohr - input argument for the Bohr radius (or base radius value) for which to build the state
#     z - number of protons in the atomic nucleus, used to determine initial velocity required for a circular orbit
#     orbital_shells - a boolean flag to determine whether or not to include orbital shells, with the default value 'False'
#                       if orbital shells are not included, state0 will be built where all electrons are in one shell with
#                       an equal angular distance between each electron. If orbital shells are included, the state0 vector will
#                       be built with the first two electrons in the first orbital shell, and all subsequent electrons in the second
#                       orbital shell. There is possible expansion for there to be more orbital shells included, however we do not 
#                       intend on observing more electrons than the first two shells can theoretically hold within the scope of this
#                       simulation
# The function returns a state0 vector properly calibrated with appropriate angular separation between each electron, with each electron
# having the same direction for angular velocity, and with the linear velocity of each electron tangential to it's initial position w.r.t.
# it's orbit around the nucleus.
def create_state0(n, r_bohr, z, r_multiplier = 1, v_multiplier = 1, orbital_shells = False):
    state0 = []
    if n < 2:
        return None
    r0 = r_multiplier * (1**2*r_bohr)/z
    v0 = v_multiplier * np.sqrt(k*z*q**2/(m*r0))
    if orbital_shells:
        state0.extend([r0, 0, 0, v0, -r0, 0, 0, -v0])
        if n > 2:
            angles = np.linspace((1/2)*np.pi, (3/2)*np.pi, n-2, endpoint=False)
            r1 = r_multiplier * (2**2*r_bohr)/z
            v1 = v_multiplier * np.sqrt(k*z*q**2/(m*r1))
            for angle in angles:
                state0.extend([r1*np.cos(angle), r1*np.sin(angle), -v1*np.sin(angle), v1*np.cos(angle)])
    else:
        if n == 2:
            state0 = [r0, 0, 0, v0, -r0, 0, 0, -v0]
        elif n == 4:
            state0 = [r0, 0, 0, v0, -r0, 0, 0, -v0, 0, r0, -v0, 0, 0, -r0, v0, 0]
        else:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            for angle in angles:
                state0.extend([r0*np.cos(angle), r0*np.sin(angle), -v0*np.sin(angle), v0*np.cos(angle)])
    return state0

# Class Saturnian_Atomic_Simulation
# This class constructs an object with several class variables
#     n                        - number of electrons for which state0 holds the information for
#     state0                   - the initial state (initial values) for each electron which will be used in solve_ivp
#                                 where each electron's state is of the form [x, y, dxdt, dydt]
#     charge                   - the value for the base particle charge, q, to be used by the system
#     fixed_charge             - the [x,y] position of the atomic nucleus
#     atomic_number            - the atomic number/number of protons (Z) in the atomic nucleus
#     mass                     - the value for mass of electron
#     min_distance             - an array of length n (one index/electron) representing the minimum distance of each electrons orbit
#     max_distance             - an array of length n (one index/electron) representing the maximum distance of each electrons orbit
#     initial_angular_position - the initial angular position of each electron within the orbit (w.r.t. +x-axis)
#     angular_position         - an array of each electron's current angular position updated as the simulation runs
#     last_angular_position    - an array of each electron's last angular position, used in conjunction with angular_position
#                                 in order to calculate the angle swept by each electron throughout it's orbit
#     angle_swept              - an array that holds information for how much of an orbit each electron has swept out; this is used to 
#                                 determine whether the values of min_distance/max_distance should be reset in order to accommodate for
#                                 changing orbits under the affects of multiple bodies
#     eccentricity_threshold   - value for eccentricity above which the electron is considered to have escaped
#     min_distance_threshold   - value for distance between electron and nucleus below which the electron is considered to have
#                                 crashed into the nucleus
#     energy_threshold         - positive value above which the electron's total energy is considered to be positive, indicating that
#                                 the electron is likely leaving a stable orbit (not currently used).
class Saturnian_Atomic_Simulation:
    def __init__(self, n, state0, charge, fixed_charge, atomic_number, mass):
        self.n = n
        self.state0 = state0
        self.charge = charge
        self.fixed_charge = fixed_charge
        self.atomic_number = atomic_number
        self.mass = mass
        self.min_distance = np.full(n, float('inf'))
        self.max_distance = np.full(n,float('-inf'))
        self.eccentricities = []
        self.initial_angular_position = np.zeros(self.n)
        for i in range(self.n):
            self.initial_angular_position[i] = np.arctan2(state0[i*4 + 1],state0[i*4])
        self.angular_position = self.initial_angular_position
        self.last_angular_position = self.initial_angular_position
        self.angle_swept = np.zeros(self.n)
        self.eccentricity_threshold = 0.95
        self.min_distance_threshold = r_bohr*10**-2
        self.energy = []
        self.energy_threshold = 1e-17
        self.sol = None

    # Class Function - calculate_eccentricity
    # The Saturnian_Atomic_Simulation class function calculate_eccentricity acts to calculate the eccentricity of each electron using
    # the formula listed in the 'Simulation' section above. The function takes input arguments
    #     r_max - an array for the max distance of each electron
    #     r_min - an array for the min distance of each electron
    # The function calculated the eccentricity and returns an array of eccentricities for each electron in the system.
    def calculate_eccentricity(self, r_max, r_min):
        return (r_max - r_min)/(r_max + r_min)

    # Class Function - calculate_energy
    # The Saturnian_Atomic_simulation class function calculate_energy acts to calculate the potential and kinetic energy of each electron
    # in the system where the kinetic energy is calculated from each electron's mass and velocity, and the potential energy is calculated
    # for each electron due to the potential from each other electron and the atomic nucleus. The function takes input arguments
    #     state - the current system state
    # and the function returns an array of length n (one index/electron) representing each electron's total energy; theoretically an electron
    # in a stable orbit should have a negative total energy. As solve_ivp is a numerical approach and there may be some loss of accuracy across
    # iterations, there is a defined non-zero positive energy threshold, above which we effectively consider the electron to have 'positive' 
    # total energy.
    def calculate_energy(self, state):

        PE_arr = np.zeros(self.n)
        KE_arr = np.zeros(self.n)
        E_arr = np.zeros(self.n)

        for i in range(self.n):
            r, r_electron = None, None
            r = np.sqrt((state[i*4]-self.fixed_charge[0])**2 + (state[i*4+1]-self.fixed_charge[1])**2)
            KE_arr[i] = 0.5 * self.mass * (state[i*4 + 2] ** 2 + state[i*4 + 3] ** 2)
            PE_arr[i] = -k * self.charge ** 2 / r
            for j in range(self.n):
                if i != j:
                    r_electron = np.sqrt((state[i*4]-state[j*4])**2 + (state[i*4+1]-state[j*4+1])**2)
                    PE_arr[i] += k * q ** 2 / r_electron
        E_arr = PE_arr + KE_arr

        return E_arr

    # Class Function - energy_event
    # The Saturnian_Atomic_Simulation class function energy_event is an event function that can be input into the event list of solve_ivp. This
    # allows for solve_ivp to monitor the total energy of each electron to see when the energy becomes effectively 'positive' and the electron
    # can be considered in an unstable orbit. As with event-type functions, energy_event takes input arguments of 't' and 'state'.
    def energy_event(self, t, state):
        E_arr = self.calculate_energy(state)
        self.energy.append(E_arr.tolist())
        unbound_bool = False
        for i in range(len(E_arr)):
            if E_arr[i] > self.energy_threshold:
                unbound_bool = True
        if unbound_bool:
            return 0
        return 1
    energy_event.terminal = True
    energy_event.direction = 0

    # Class Function - eccentricity_event
    # The Saturnian_Atomic_Simulation class function eccentricity_event is an event function that can be input into the event list of solve_ivp. This
    # function allows for solve_ivp to monitor the eccentricity of each electron and stops the program when the eccentricity of any electron
    # passes above a threshold, at which the electron is considered unbound from it's orbit.
    # This event calls on the class function calculate_eccentricity in order to determine each electron's eccentricity. Furthermore, if the 
    # eccentricity is still below the threshold and the electron sweeps out an angle of at least 2*pi (completes one orbit around the nucleus) the
    # method resets the value of that electrons r_min/r_max - this is done such that if an electron is experiencing a highly elliptical orbit where
    # due to the affects of other orbiting bodies the minimum distance increases (or maximum distance decreases) that the electron doesn't hold on
    # to values it no longers passes through
    def eccentricity_event(self, t, state):

        eccentricity_bool = False
        e = self.calculate_eccentricity(self.max_distance, self.min_distance)
        self.eccentricities.append(e.tolist())

        for i in range(self.n):
            self.angular_position[i] = np.arctan2(state[i*4+1],state[i*4])

        angular_difference = np.abs(self.angular_position - self.last_angular_position)

        # Normalize the angular difference to the range [-π, π]
        angular_difference = (angular_difference + np.pi) % (2 * np.pi) - np.pi

        self.angle_swept += np.abs(angular_difference)
        self.last_angular_position = self.angular_position

        for i in range(self.n):
            if e[i] >= self.eccentricity_threshold:
                print(f"Simulation ending at t={t}\nElectron {i+1} unstable with eccentricity {e[i]}")
                eccentricity_bool = True

        if eccentricity_bool:
            return 0

        for i in range(self.n):
            if self.angle_swept[i] > 2 * np.pi:
                self.initial_angular_position[i] = self.angular_position[i]
                self.angle_swept[i] = 0
                self.min_distance[i] = float('inf')
                self.max_distance[i] = float('-inf')

        return 1
    eccentricity_event.terminal = True
    eccentricity_event.direction = 0

    # Class Function - crash_event
    # The Saturnian_Atomic_Simulation class function crash_event is an event function that calculates the distance of each electron from the
    # atomic nucleus and calls for the program to stop of any of the distances falls below a certain threshold, at which point the electron
    # is considered to have crashed into the atomic nucleus.
    # The way which this simulation calculates the force on an electron due to the nucleus, if the distance becomes very small the force
    # on the electron will become extremely large due to the force being an Inverse-Square force; this minimum threshold stops the program
    # when this is considered to have occured
    def crash_event(self, t, state):
        # Calculate distances of each electron to the proton (fixed charge)
        r_arr = np.zeros(self.n)
        crash_bool = False
        for i in range(self.n):
            if np.sqrt((state[i*4]-self.fixed_charge[0])**2 + (state[i*4+1]-self.fixed_charge[1])**2) <= self.min_distance_threshold:
                print(f"Crash event triggered at time {t} for electron {i+1}.")
                crash_bool = True
        if crash_bool:
            return 0
        return 1
    crash_event.terminal = True
    crash_event.direction = 0

    # Class Function - diff_equation
    # The Saturnian_Atomic_Simulation class function diff_equation represents the differential equation that solve_ivp uses to simulate our
    # Saturnian Atomic Model.
    # For each electron within the initial state0 vector (n = len(state0)%4) the differential equation calculates the force on said electron
    # due to: (1) the atomic nucleus; and (2) each other electron within orbit. This is done in accordance with the listed theories above.
    # The diff_equation function also updates each electrons r_max/r_min from which eccentricity of the particle may be calculated and stability
    # determined.
    # The way that this function was written was such that a state vector of arbitrary length may be passed into the differential equation
    # and solve_ivp; this allows for the model to effectively model anywhere from 1 to an arbitrary number of electrons (although the 
    # emphasis of this simulation will focus on two electrons for the majority of the phase-space analysis, with some attention being spent
    # more electrons in a second orbital shell.
    def diff_equation(self, t, state):

        next_state = np.zeros(len(state))

        for i in range(self.n):
            fx, fy = 0 ,0
            r, r_electron = None, None

            # Calculate the distance between the electron and the nucleus
            r = np.sqrt(state[i*4]**2 + state[i*4+1]**2)

            if r > self.max_distance[i]:
                self.max_distance[i] = r
            if r < self.min_distance[i]:
                self.min_distance[i] = r

            # Calculate forces on the electron from the nucleus
            if r > 0:
                fx += -1*(k*self.atomic_number*self.charge**2/r**2)*(state[i*4]/r)
                fy += -1*(k*self.atomic_number*self.charge**2/r**2)*(state[i*4+1]/r)

            for j in range(self.n):
                if i != j:
                    dx = state[i*4]-state[j*4]
                    dy = state[i*4+1]-state[j*4+1]
                    r_electron = np.sqrt(dx**2 + dy**2)
                    if r_electron > 0:
                        fx += (k*self.charge**2/r_electron**2)*(dx/r_electron)
                        fy += (k*self.charge**2/r_electron**2)*(dy/r_electron)

            next_state[i*4], next_state[i*4+1], next_state[i*4+2], next_state[i*4+3] = state[i*4+2], state[i*4+3], fx/self.mass, fy/self.mass

        return next_state

    # Class Function - plot_trajectories
    # The Saturnian_Atomic_Simulation class function plot_trajectories takes an input argument
    #     sol - solution array output from solve_ivp
    # and from that information it creates a subplot for the trajectories of each electron around the atomic nucleus.
    def plot_trajectories(self, zoom_factor):
        t_sol = self.sol.t
        y_sol = self.sol.y

        x_fixed_charge, y_fixed_charge = self.fixed_charge

        fig, ax = plt.subplots(1, 2, figsize=(16, 12))

        ax[0].scatter(x_fixed_charge, y_fixed_charge, color='b', label="Nucleus")
        ax[1].scatter(x_fixed_charge, y_fixed_charge, color='b', label="Nucleus")

        for i in range(self.n):
            ax[0].plot(self.state0[i*4], self.state0[i*4+1], color=colours[i], marker='x', markersize=10)
            ax[0].plot(y_sol[i*4], y_sol[i*4+1], color=colours[i], linestyle=line_styles[i % len(line_styles)], label=f"Electron {i+1} Trajectory")
            ax[1].plot(self.state0[i * 4], self.state0[i * 4 + 1], color=colours[i], marker='x', markersize=10)
            ax[1].plot(y_sol[i * 4], y_sol[i * 4 + 1], color=colours[i], linestyle=line_styles[i % len(line_styles)], label=f"Electron {i + 1} Trajectory")

        ax[0].set_aspect('equal', 'box')
        xlim = ax[0].get_xlim()
        ylim = ax[0].get_ylim()
        max_range = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])) / 2.0
        ax[0].set_xlim(-max_range, max_range)
        ax[0].set_ylim(-max_range, max_range)
        ax[0].set_xlabel("x [m]")
        ax[0].set_ylabel("y [m]")
        ax[0].set_title("Electron Trajectories")
        ax[0].grid(True)
        ax[0].legend(loc='upper left')

        ax[1].set_aspect('equal', 'box')
        ax[1].set_xlabel("x [m]")
        ax[1].set_ylabel("y [m]")
        zoom_range = zoom_factor * r_bohr
        ax[1].set_xlim(self.fixed_charge[0] - zoom_range, self.fixed_charge[0] + zoom_range)
        ax[1].set_ylim(self.fixed_charge[1] - zoom_range, self.fixed_charge[1] + zoom_range)
        ax[1].set_title(f"Electron Trajectories (Scaled to {zoom_factor} times Bohr Radius)")
        ax[1].grid(True)
        ax[1].legend(loc='upper left')
        plt.show()

    # Class Function - plot_stability
    # The Saturnian_Atomic_Simulation class function plot_stability plots both the eccentricity and total energy of each electron as a
    # function of time, where the eccentricities are calculated within the eccentricity_event event function and appended to a list of eccentricities
    # and the energies are calculated within the energy_event event function and appended to a list of energies within the class variables.
    # This class first ensures that the length of solve_ivp sol.t is the same as the length of the energies/eccentricities list, and if not
    # it forces the same size such that the data can be plotted - this involves either trimming the length of energies/eccentricities
    # or by padding the end of energies/eccentricities with the last recorded values such that they are both the same length as sol.t
    def plot_stability(self):
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 12))

        if len(self.eccentricities) > len(self.sol.t):
            self.eccentricities = self.eccentricities[:len(self.sol.t)]
        elif len(self.eccentricities) < len(self.sol.t):
            last_eccentricity = self.eccentricities[-1]
            self.eccentricities.extend([last_eccentricity] * (len(self.sol.t) - len(self.eccentricities)))

        if len(self.energy) > len(self.sol.t):
            self.energy = self.energy[:len(self.sol.t)]
        elif len(self.energy) < len(self.sol.t):
            last_energy = self.energy[-1]
            self.energy.extend([last_energy] * (len(self.sol.t) - len(self.energy)))

        ax[0].axhline(y=self.eccentricity_threshold, color='k', linestyle='-', linewidth=1.5, label=f"Eccentricity Threshold, e = {self.eccentricity_threshold}")
        for i in range(self.n):
            ax[0].plot(self.sol.t, [ecc[i] for ecc in self.eccentricities], label=f'Electron {i+1}', color=colours[i], linestyle=line_styles[i % len(line_styles)])
        ax[0].set_xlabel('Time, t [s]')
        ax[0].set_ylabel('Eccentricity')
        ax[0].set_title('Eccentricity of Electrons Over Time')
        ax[0].legend()
        ax[0].grid(True)
    
        ax[1].axhline(y=self.energy_threshold, color='k', linestyle='-', linewidth=1.5, label=f"Energy Threshold for Bound Electron, E = {self.energy_threshold} J")
        for i in range(self.n):
            ax[1].plot(self.sol.t, [energy[i] for energy in self.energy], label=f'Electron {i+1}', color=colours[i], linestyle=line_styles[i % len(line_styles)])
        ax[1].set_xlabel('Time, t [s]')
        ax[1].set_ylabel('Energy, E [J]')
        ax[1].set_title('Energy of Electrons Over Time')
        ax[1].legend()
        ax[1].grid(True)
    
        plt.tight_layout()
        plt.show()

    

    # Class Function - plot_eccentricities
    # The Saturnian_Atomic_Simulation class function plot_eccentricities plots the eccentricity of each electron as a function of time
    # where the eccentricities are calculated within the eccentricity_event event function and appended to a list of eccentricities within
    # the Saturnian_Atomic_Simulation class.
    # This class first ensures that the length of solve_ivp sol.t is the same as the length of the eccentricities list, and if not
    # it forces the same size such that the data can be plotted - this involves either trimming the length of eccentricities such that its
    # the same length as sol.t, or by padding the end of eccentricities with the last recorded value such that its the same length as sol.t
    # NOTE: The functionality of this class function has been absorbed into the umbrella function 'plot_stability' such that the plot produced
    #        by this function can be displayed as 1 of 2 subplots (with the other being total electron energy vs time). This function has been
    #        left within the class definition such that it may be called individually if desired.
    def plot_eccentricities(self):
        
        if len(self.eccentricities) > len(self.sol.t):
            self.eccentricities = self.eccentricities[:len(self.sol.t)]
        elif len(self.eccentricities) < len(self.sol.t):
            last_eccentricity = self.eccentricities[-1]
            self.eccentricities.extend([last_eccentricity] * (len(self.sol.t) - len(self.eccentricities)))
            
        plt.figure(figsize=(16, 12))
        plt.axhline(y=self.eccentricity_threshold, color='k', linestyle='-', linewidth=1.5, label=f"Eccentricity Threshold, e = {self.eccentricity_threshold}")
        for i in range(self.n):
            plt.plot(self.sol.t, [ecc[i] for ecc in self.eccentricities], label=f'Electron {i+1}', color=colours[i], linestyle=line_styles[i % len(line_styles)])
        plt.xlabel('Time, t [s]')
        plt.ylabel('Eccentricity')
        plt.title('Eccentricity of Electrons Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Class Function - plot_energies
    # The Saturnian_Atomic_Simulation class function plot_energies plots the total energy of each electron as a function of time
    # where the energies are calculated within the energy_event event function and appended to a list of electron energies within
    # the Saturnian_Atomic_Simulation class.
    # This class first ensures that the length of solve_ivp sol.t is the same as the length of the energies list, and if not
    # it forces the same size such that the data can be plotted - this involves either trimming the length of energies such that its
    # the same length as sol.t, or by padding the end of energies with the last recorded values such that its the same length as sol.t
    # NOTE: The functionality of this class function has been absorbed into the umbrella function 'plot_stability' such that the plot produced
    #        by this function can be displayed as 1 of 2 subplots (with the other being electron eccentricity vs time). This function has been
    #        left within the class definition such that it may be called individually if desired.
    def plot_energies(self):
        
        if len(self.energy) > len(self.sol.t):
            self.energy = self.energy[:len(self.sol.t)]
        elif len(self.energy) < len(self.sol.t):
            last_energy = self.energy[-1]
            self.energy.extend([last_energy] * (len(self.sol.t) - len(self.energy)))

        plt.figure(figsize=(16, 12))
        plt.axhline(y=self.energy_threshold, color='k', linestyle='-', linewidth=1.5, label=f"Energy Threshold for Bound Electron, E = {self.energy_threshold} J")
        for i in range(self.n):
            plt.plot(self.sol.t, [energy[i] for energy in self.energy], label=f'Electron {i+1}', color=colours[i], linestyle=line_styles[i % len(line_styles)])
        
        plt.xlabel('Time, t [s]')
        plt.ylabel('Energy, E [J}')
        plt.title('Energy of Electrons Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Class Function - run_simulation
    # The Saturnian_Atomic_Simulation class function run_simulation is the function that executes and runs the simulation for the class
    # instance. The class function accepts the following input arguments:
    #     t0       - start time of the simulation
    #     tmax     - max run time of the simulation; this argument is best input as some natural number multiple of the expected
    #                 orbital period of the electrons depending on the initial conditions given in state0
    #     max_step - the maximum allowed timestep that solve_ivp is allowed to take when running the simulation; again, this argument is best
    #                 input as a magnitude below the expected orbital period (ie T*10^-3 such that there are 1000 steps for each orbital period)
    #     atol     - absolute tolerance in error allowed during each iteration of solve_ivp
    #     rtol     - relative tolerance in error allowed during each iteration of solve_ivp
    # This function runs scipy.integrate.solve_ivp with the values given in the input arguments and stores the solution in the class variable
    # self.sol such that the solution can then be called by other class functions to plot data.
    def run_simulation(self, t0=0, tmax=1, max_step=1e-6, atol=1e-6, rtol=1e-3):
        t_span = (t0, tmax)
        sol = solve_ivp(lambda t, state:self.diff_equation(t, state),
                        t_span,
                        self.state0,
                        max_step = max_step,
                        atol = atol,
                        rtol = rtol,
                        events = [self.eccentricity_event, self.crash_event, self.energy_event]
        )
        self.sol = sol
        
        if sol.success:
            # Check if an event triggered
            if sol.t_events[0].size > 0:
                print(f"Simulation ended due to eccentricity_event triggering at time {sol.t_events[0]}")
            elif sol.t_events[1].size > 0:
                print(f"Simulation ended due to crash_event triggering at time {sol.t_events[1]}")
            elif sol.t_events[2].size > 0:
                print(f"Simulation ended due to energy_event triggering at time {sol.t_events[2]}")
            else:
                print("Simulation ended due to reaching maximum time.")
        else:
            print("Simulation failed.")

print("System Initialized")