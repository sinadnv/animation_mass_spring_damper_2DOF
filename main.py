# THE FOLLOWING CODE USED RUNGE KUTTA FOURTH ORDER METHOD TO CALCULATE THE POSITION AND VELOCITY OF A 2-DOF SYSTEM.
# FIRST, THE NATURAL FREQUENCY, MODE SHAPE, AND MASS PARTICIPATION FACTOR ARE CALCULATED USING SCIPY PACKAGE.
# SECOND, THE RESPONSE OF THE SYSTEM GIVEN THE INITIAL CONDITIONS AS WELL AS THE FORCE INPUT IS CALCULATED.
# THE CODE CALCULATES THE RESPONSE AT THE FUNDAMENTAL FREQUENCY. CHANGE THE FUNCTION INPUT TO SEE THE RESPONSE AT OTHER
# FREQUENCIES.
# THIRD AND LAST, THE RESPONSE OF THE SYSTEM IS ANIMATED USING THE PYGAME PACKAGE.
# ################ COPYRIGHT SINA DANESHVAR --- NOVEMBER 20, 2021 ################
# DO NOT USE WITHOUT ACKNOWLEDGEMENT
## Start
import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
from scipy.linalg import eigh
from pygame.locals import *

# Set up variables. Change accordingly
# Defining system's properties
# m: mass, k: stiffness, c: damping
# F0: force amplitude
m1, m2 = 1, 1
k1, k2 = 2, 1
k3 = 0
c1, c2 = 1, 1
F0 = 10

# Define time
tmax = 50
h = .01
time = np.arange(0, tmax, h)

# Initial values at t = 0
x1 = 0
u1 = 0
x2 = 0
u2 = 0
pos1 = [x1]     # list of mass 1 positions over time
vel1 = [u1]     # list of mass 1 velocities over time
pos2 = [x2]     # list of mass 2 positions over time
vel2 = [u2]     # list of mass 2 velocities over time


# Solve for eigenvalues (natural frequencies) of the system
# Solve for eigenvectors (mode shapes) of the system
# Matrix to solve: K-w^2*M
K = np.array([[k1+k2, -k2], [-k2, k3+k2]])
M = np.array([[m1, 0], [0, m2]])

f, v = eigh(K, M)
f = np.sqrt(f)
mass_contribution = np.matmul(np.matmul(np.transpose(v), M), np.array([[1], [1]]))
print('The Natural Frequencies are: ', np.round(f[0], 2), 'and', np.round(f[1], 2), 'rad/s')
print('The EigenVectors are: \n', v)

# Read the natural frequencies to see the system's response at resonance
omega1, omega2 = f[0], f[1]
# Change the frequency of interest below
omega = omega2

## Solve for positions of m1 and m2
def force(t, omega):
    return F0*np.sin(omega*t)


# In Runge Kutta, y' = f(y,t). So I calculated the derivatives of each state as a function of others.
# x[0] = mass 1 position, x[0] = mass 1 velocity, x[2] = mass 2 position, x[3] = mass 2 velocity
# The RK function assumes the harmonic frequency is the fundamental natural frequency of the system. Change if you wish
# to see the response in other frequencies.
def dxdt(x, t, omega = omega1):
    du1dt = x[1]
    du2dt = (-(c1+c2)*x[1]+c2*x[3]-(k1+k2)*x[0]+k2*x[2])/m1
    du3dt = x[3]
    du4dt = (c2*x[1]-c2*x[3]+k2*x[0]-k2*x[2]+force(t, omega))/m2
    return np.array([du1dt, du2dt, du3dt, du4dt])


# Dumping all initial values in a list to be used in the for loop to be accessible to the dxdt function
x = [x1, u1, x2, u2]
# Starting the for loop from the second array as the first array corresponds to t = 0 which its information is
# already given (initial values).
# Runge Kutta functions are defined at each step to calculate the updated the value of x
# each array of x is appended to their corresponding list for plotting
for t in time[1:,]:
    RK1 = h*dxdt(x,t)
    RK2 = h*dxdt(x+RK1/2, t+h/2)
    RK3 = h*dxdt(x+RK2/2, t+h/2)
    RK4 = h*dxdt(x+RK3  , t+h)

    x += (RK1+2*RK2+2*RK3+RK4)/6

    pos1.append(x[0])
    vel1.append(x[1])
    pos2.append(x[2])
    vel2.append(x[3])
# Velocity data can be access too, if required

# Plotting positions
plt.plot(time, pos1)
plt.plot(time, pos2)
legend = ['pos1', 'pos2']
plt.legend(legend)
plt.grid()
plt.show()


pos1 = np.array(pos1)
pos2 = np.array(pos2)
plt.plot(time, pos1-pos2)
plt.show()

## Draw the blocks using Pygame
# I use an object-oriented approach to define the mass blocks
# The following variables are set for a better demo and may need to be changed for other mass and spring configurations
amp_factor = 5  # Amplify the displacement by this factor to make it more discernible
mass1_origin = 150  # Starting position of M1
mass2_origin = 300  # Starting position of M2
x_origin = 20  # Wall origin (x)
y_origin = 350  # Wall origin (y)
Xmax = 400  # Length of the display window
Ymax = 400  # Height of display window

# Initialize pygame
pygame.init()

# To draw the static wall
def draw_wall():
    pygame.draw.line(screen, 'BLACK', (x_origin, y_origin), (x_origin, y_origin-80),width=3)
    pygame.draw.line(screen, 'BLACK', (x_origin, y_origin), (x_origin+350, y_origin),width=3)


def display_time(time, Xloc, Yloc, font=pygame.font.SysFont(pygame.font.get_default_font(), 32)):
    # default_font = pygame.font.get_default_font()
    # font = pygame.font.SysFont(default_font, 32)
    string = r't = {t} s'.format(t=np.round(time, 2))
    text = font.render(string, True, 'RED')
    textRect = text.get_rect()
    textRect.center = (Xloc, Yloc)
    screen.blit(text, textRect)


def display_config(Xloc, Yloc, font=pygame.font.SysFont(pygame.font.get_default_font(), 20)):
    string1 = r'm1 = {m}'.format(m=m1)
    string2 = r'm2 = {m}'.format(m=m2)
    string3 = r'k1 = {k}'.format(k=k1)
    string4 = r'k2 = {k}'.format(k=k2)
    string5 = r'c1 = {c}'.format(c=c1)
    string6 = r'c2 = {c}'.format(c=c2)
    text1 = font.render(string1, True, 'BLACK')
    text2 = font.render(string2, True, 'BLACK')
    text3 = font.render(string3, True, 'BLACK')
    text4 = font.render(string4, True, 'BLACK')
    text5 = font.render(string5, True, 'BLACK')
    text6 = font.render(string6, True, 'BLACK')

    textRect = text1.get_rect()
    textRect.center = (Xloc, Yloc)
    screen.blit(text1, textRect)

    textRect = text2.get_rect()
    textRect.center = (Xloc, Yloc+15)
    screen.blit(text2, textRect)

    textRect = text3.get_rect()
    textRect.center = (Xloc, Yloc+30)
    screen.blit(text3, textRect)

    textRect = text4.get_rect()
    textRect.center = (Xloc, Yloc+45)
    screen.blit(text4, textRect)

    textRect = text5.get_rect()
    textRect.center = (Xloc, Yloc+60)
    screen.blit(text5, textRect)

    textRect = text6.get_rect()
    textRect.center = (Xloc, Yloc+75)
    screen.blit(text6, textRect)


# Object mass. To be assigned to each block
class Mass:
    def __init__(self, position, width = 60, height = 40):
        self.pos = position
        self.w = width
        self.h = height
        self.left = self.pos-self.w/2
        self.right = self.pos+self.w/2
        self.top = y_origin-60+self.h/2

    # Render mass
    def show(self):
        pygame.draw.rect(screen, 'BLUE', (self.left, self.top, self.w, self.h))

    # update the mass position
    def update_position(self, position):
        self.pos = position
        self.left = self.pos-self.w/2
        self.top = y_origin-60+self.h/2
        self.right = self.pos+self.w/2


# Object Spring. To be assigned to each spring
class Spring:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.height = y_origin-30
        self.length = self.end-self.start

    # render the spring
    def show(self):
        pygame.draw.lines(screen,'BLACK',False, [(self.start, self.height),
                                                 (self.start + 0.02 * self.length, self.height),
                                                 (self.start + 0.06 * self.length, self.height + 0.08 * self.length),
                                                 (self.start + 0.14 * self.length, self.height - 0.08 * self.length),
                                                 (self.start + 0.22 * self.length, self.height + 0.08 * self.length),
                                                 (self.start + 0.30 * self.length, self.height - 0.08 * self.length),
                                                 (self.start + 0.38 * self.length, self.height + 0.08 * self.length),
                                                 (self.start + 0.46 * self.length, self.height - 0.08 * self.length),
                                                 (self.start + 0.54 * self.length, self.height + 0.08 * self.length),
                                                 (self.start + 0.62 * self.length, self.height - 0.08 * self.length),
                                                 (self.start + 0.70 * self.length, self.height + 0.08 * self.length),
                                                 (self.start + 0.78 * self.length, self.height - 0.08 * self.length),
                                                 (self.start + 0.86 * self.length, self.height + 0.08 * self.length),
                                                 (self.start + 0.94 * self.length, self.height - 0.08 * self.length),
                                                 (self.start + 0.98 * self.length, self.height),
                                                 (self.end, self.height)], width=2)

    # update the spring start and end nodes
    def update(self, start, end):
        self.start = start
        self.end = end
        self.length = self.end - self.start


# Create mass and spring objects
mass1 = Mass(mass1_origin)
mass2 = Mass(mass2_origin)
spring1 = Spring(x_origin, mass1.left)
spring2 = Spring(mass1.right, mass2.left)

# Set up initial pygame variables
clock = pygame.time.Clock()
screen = pygame.display.set_mode((Xmax, Ymax))
screen.fill('WHITE')
pygame.display.flip()
pygame.display.set_caption(r'F = {F}sin({w}t)'.format(F=F0, w=np.round(omega, 2)))

# Read the time every 10 array
for counter in range(0, len(time), 10):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    # Show static wall, mass blocks, and springs; update the screen so that the changes take place
    display_time(time[counter],50,50)
    display_config(Xmax-50, 20)
    draw_wall()
    mass1.show()
    mass2.show()
    spring1.show()
    spring2.show()
    pygame.display.update()

    screen.fill('WHITE')  # clear the screen to get ready for the next position

    # Update the location of the mass blocks and springs to get ready for the next iteration
    mass1.update_position((mass1_origin + amp_factor * pos1[counter]))
    mass2.update_position((mass2_origin + amp_factor * pos2[counter]))
    spring1.update(x_origin, mass1.left)
    spring2.update(mass1.right, mass2.left)

    clock.tick(10)  # frame rate

