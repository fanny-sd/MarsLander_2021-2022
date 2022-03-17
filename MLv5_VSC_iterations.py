# V5: Drag in Horz direction + height by X[1] change + fuel + PID + 2D + Drag
#CHANGES:
# Ki  and Kd passed 
# ADDED JUST TO VSC
    # OutofFuel = 0
    # pass 'fuel_warning_printed' in simulation

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive  # slider
from matplotlib import rcParams 
from numpy.linalg import norm
from numpy.random import randint
from ipywidgets import interactive 
from matplotlib import rcParams  

rcParams['figure.figsize'] = (10, 8)

def mars_surface():
    surfaceN = randint(5, 15)
    land = np.zeros((surfaceN, 2), dtype=int)
    
    # ensure there's a flat landing site at least 1000m long
    landing_site = randint(1, surfaceN-1)
    land[landing_site, 0] = randint(2000, 5000)
    land[landing_site+1, 0] = min(land[landing_site, 0] + randint(1000, 2000), 6999)
    land[landing_site+1, 1] = land[landing_site, 1] = randint(1, 1500)
    
    # fill in the rest of the terrain
    for i in range(landing_site):
        land[i, 0] = (land[landing_site, 0] / landing_site) * i
        land[i, 1] = randint(0, 1500)
    
    for i in range(landing_site + 2, surfaceN):
        land[i, 0] = (land[landing_site + 1, 0] + 
                      (7000 - land[landing_site + 1, 0]) / len(land[landing_site + 2:]) * 
                      (i - (landing_site + 1)))
        land[i, 1] = randint(0, 1500)
    
    # impose boundary conditions
    land[0, 0] = 0
    land[-1, 0] = 6999

    return land, landing_site

def plot_surface(land, landing_site):
    fig, ax = plt.subplots()
    ax.plot(land[:landing_site+1, 0], land[:landing_site+1, 1], 'k-')
    ax.plot(land[landing_site+1:, 0], land[landing_site+1:, 1], 'k-')
    ax.plot([land[landing_site, 0], land[landing_site+1, 0]], 
             [land[landing_site, 1], land[landing_site+1, 1]], 'k--')
    ax.set_xlim(0, 7000)
    ax.set_ylim(0, 16000)
    return ax

def plot_lander(land, landing_site, X, thrust=None, animate=False, step=10):
    if animate:
        def plot_frame(n=len(X)-1):
            ax = plot_surface(land, landing_site)
            ax.plot(X[:n, 0], X[:n, 1], 'b--')      # trajectory of lander
            ax.plot(X[n, 0], X[n, 1], 'k^', ms=20)
            if thrust is not None:
                ax.plot([X[n, 0], X[n, 0] - 100*thrust[n, 0]],
                        [X[n, 1] - 100., X[n, 1] - 100. - 100*thrust[n, 1]], 
                       'r-', lw=10)
        return interactive(plot_frame, n=(0, len(X), step)) #slider
    else:
        ax = plot_surface(land, landing_site) 
        ax.plot(X[:, 0], X[:, 1], 'b--')
        ax.plot(X[-1, 0], X[-1, 1], 'b^')
        return ax

def interpolate_surface(land, x):
    i,  = np.argwhere(land[:, 0] < x)[-1] # segment containing x is [i, i+1]
    m = (land[i+1, 1] - land[i, 1])/(land[i+1, 0] - land[i, 0]) # gradient
    x1, y1 = land[i, :] # point on line with eqn. y - y1 = m(x - x1) 
    return m*(x - x1) + y1

land, landing_site = mars_surface()

def height(land, X):
    return X[1] - interpolate_surface(land, X[0]) #1 in X[1] points to the vertical position y of the lander

assert abs(height(land, [1, land[0, 1]])) < 100.0 # height when on surface left edge should be close to zero
assert abs(height(land, [6999, land[-1, 1]])) < 100.0 # height when on surface at right edge should be close to zero

_land, _landing_site = mars_surface()

def _height(_land, X):
    return X[1] - interpolate_surface(_land, X[0])

points = np.zeros((10, 2))
points[:, 0] = randint(0, 7000, size=10)
points[:, 1] = randint(0, 16000, size=10)
for i in range(10):
    assert abs(height(_land, points[i, :]) - _height(_land, points[i, :])) < 1e-6


g = 3.711 # m/s^2 , gravity on Mars
TSFC = 0.0003 # kg/(N*s)
Dc = 6.3525 # drag force as a function of velocity

def simulate(X0, V0, land, landing_site, 
             fuel=400, dt=0.1, Nstep=1000, 
             autopilot=None, print_interval=100, parameters=None, parachute=None):
    
    n = len(X0)       # number of degrees of freedom (2 here)
    X = X0.copy()     # current position
    V = V0.copy()     # current velocity
    Xs = np.zeros((Nstep, n)) # position history (trajectory) 
    Vs = np.zeros((Nstep, n)) # velocity history
    thrust = np.zeros((Nstep, n)) # thrust history
    drag = np.zeros((Nstep, n)) # drag history
    success = False
    fuel_warning_printed = False
    rotate = randint(-90, 90)   # degrees, random initial angle random
    power = 0    # m/s^2, initial thrust power  
    
    e_prev = np.zeros(Nstep) # error history   

    for i in range(Nstep):
        Xs[i, :] = X     # Store positions
        Vs[i, :] = V     # Store velocities
        
        if autopilot is not None:
            
            rotate, power, parachute = autopilot(i, X, V, fuel, rotate, power, parameters, parachute,dt,e_prev,Nstep)
            assert abs(rotate) <= 90
            assert 0 <= power <= 12000
        
            rotate_rad = rotate * np.pi / 180.0 # degrees to radians
            thrust[i, :] = power * np.array([np.sin(rotate_rad), 
                                             np.cos(rotate_rad)])
            if fuel <= 0: 
                if not fuel_warning_printed:
                    print("Fuel empty! Setting thrust to zero")
                    fuel_warning_printed = True
                thrust[i, :] = 0
            else:
                fuel -= TSFC * power * dt
                
        m = 2600 + fuel  #kg , Mass of Lander + Rover + fuel  # fuel Mass loss
        
        if parachute == 0:
            # no Drag
            drag[i, :] = 0
        else: # parachute == 1
            # Drag - Parachute deployed
            drag[i, :] = -Dc*np.linalg.norm(V)*V
        
        A = np.array([0, -g]) + thrust[i, :]/m + drag[i, :]/m
                                   
        V += A * dt                          # update velocities
        X += V * dt                          # update positions
        
        #if i % print_interval == 0: 
            #print(f"i={i:03d} X=[{X[0]:8.3f} {X[1]:8.3f}] V=[{V[0]:8.3f} {V[1]:8.3f}]"
                  ##f" thrust=[{thrust[i, 0]:8.3f} {thrust[i, 1]:8.3f}] fuel={fuel:8.3f} rotate={rotate:8.3f} parachute={parachute:8.3f}") 
        
        # check for safe or crash landing
        if X[1] < interpolate_surface(land, X[0]):
            if not (land[landing_site, 0] <= X[0] and X[0] <= land[landing_site + 1, 0]):
                print("crash! did not land on flat ground!")
            elif rotate != 0:
                print("crash! did not land in a vertical position (tilt angle = 0 degrees)")
            elif abs(V[1]) >= 27: #was 40
                print("crash! vertical speed must be limited (<27m/s in absolute value), got ", abs(V[1]))
            elif abs(V[0]) >= 5: #was 20
                print("crash! horizontal speed must be limited (<5m/s in absolute value), got ", abs(V[0]))
            else:
                if fuel_warning_printed == True :
                    print('landed but no fuel remaining!')
                else:
                    print("safe landing - well done!")
                    success = True
            Nstep = i
            break
    
    return Xs[:Nstep,:], Vs[:Nstep,:], thrust[:Nstep,:], success,fuel_warning_printed, fuel, rotate, parachute

def pid_autopilot(i, X, V, fuel, rotate, power,parameters, parachute, dt, e_prev, Nstep):
    K_v,K_p,K_h,K_i,K_d = parameters
    
    c_v = 10.0 # target landing speed in vertical direction (m/s)
    c_h = 0 # target landing speed in horizontal direction (m/s)
    
    # Height from landing platform
    h = X[1]-land[landing_site, 1]
    
    # Horizontal displacement
    Xtarget = (land[landing_site+1, 0] + land[landing_site, 0]) // 2 
    dist = (Xtarget-X[0])

    rotate = np.rad2deg(np.arctan2(dist,h-2000))   
    
    # Ensuring the shuttle lands in a vertical position:
    if h<2000:
        rotate = 0
      
    # Combine vertical & horizontal errors
    v_target_vert = -(c_v + K_v*(h-2000))
    v_target_horz = abs(c_h+K_h*dist)
    v_err_vert = abs(v_target_vert - V[1])
    v_err_horz = abs(v_target_horz - V[0])
    e =  v_err_vert + v_err_horz
    
    e_d = 0
    if i>0:
        e_d = K_d*((e - e_prev[i-1])/dt)
        
    e_prev[i] = e     # Store error
    
    Pout = K_p*(e + e_d + K_i*(e_prev.sum()*dt)) 
    
    power = min(max(Pout, 0.0), 12000.0)   # max thrust
    
    if h > 10000:
        parachute = 0 
    else:
        parachute = 1 #open parachute
    
    #if i % 100 == 0:
        #print(f'e={e:8.3f} Pout={Pout:8.3f} power={power:8.3f} K_p={K_p:8.3f} K_h={K_h:8.3f} K_v={K_v:8.3f} K_i={K_i:8.3f} K_d={K_d:8.3f}')     
    return (rotate, power, parachute)

# TESTING

#np.random.seed(122) # seed random number generator for reproducible results
land, landing_site = mars_surface()

# PID iterations test:
#K_p= 50.000 ; K_h= 0.005 ; K_v= 0.005 ; K_i= 0.005 ; K_d= 75.000  # GREAT RESULTS
#K_p= 50.000; K_h= 0.010 ; K_v= 0.005 ; K_i= 0.005 ; K_d= 75.000  # V GOOD
#K_p= 50.000 ; K_h= 0.005; K_v= 0.010 ;K_i= 0.005; K_d= 75.000 # BAD

# P iterations test:
#K_p= 200; K_h= 0.002; K_v= 0.006; K_i= 0 ; K_d= 0 # V GOOD
K_p= 210; K_h= 0.002; K_v= 0.006; K_i= 0 ; K_d= 0 # GREAT results

OutofBounds= 0
OutofFuel = 0
trials = 0
count = 0

iterations = 200
for i in list(range(iterations)):
    X0 = [randint(2000, 5000), randint(15000, 16000)] 
    V0 = [randint(-50,50), randint(-500,-300)]
    try:
        Xs, Vs, thrust, success,fuel_warning_printed, fuel, rotate, parachute = simulate(X0, V0, land, landing_site, dt=0.1, Nstep=3500, 
                                        autopilot=pid_autopilot, fuel=400,parameters=[K_v,K_p,K_h,K_i,K_d],parachute=None)
    except IndexError:
        print('Error: Out of bounds')
        OutofBounds += 1
        continue
    count += success
    OutofFuel += fuel_warning_printed
    trials += 1

print('Number of success (with fuel remaining):',count,'out of',iterations)
print('Number of Out of fuel:', OutofFuel)
print('Number of Out of bound errors:', OutofBounds)
print('Number of trials that ran without error:',trials)  
print('Success rate:',(count/(trials+OutofBounds))*100)

plot_lander(land, landing_site, Xs, thrust, animate=True, step=10)

