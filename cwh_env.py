import gymnasium as gym
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from control import lqr
import scipy
from scipy.integrate import solve_ivp
from config import TrainConfig

def eom_cwh(t, X, Xv, K, n):
    # A=np.concatenate((np.concatenate((np.zeros((3,3)),np.eye(3)), axis=1),
    #                   np.concatenate((np.array([[3*n**2,0,0],[0,0,0],[0,0,-n**2]]),np.array([[0,2*n,0],[-2*n,0,0],[0,0,0]])), axis=1)))
    # B=np.concatenate((np.zeros((3,3)),np.eye(3)))
    A,B=dynamics(n)
    u = K@(X-Xv)
    dX = A@X + B@u
    return dX

def dynamics(n):
    Ac=np.concatenate((np.concatenate((np.zeros((3,3)),np.eye(3)), axis=1),
                      np.concatenate((np.array([[3*n**2,0,0],[0,0,0],[0,0,-n**2]]),np.array([[0,2*n,0],[-2*n,0,0],[0,0,0]])), axis=1)))
    Bc=np.concatenate((np.zeros((3,3)),np.eye(3)))
    return Ac, Bc

def DT_dynamics(n,dt):
    nt=n*dt
    Ad_rr=np.array([[4-3*np.cos(nt),0,0],[6*(np.sin(nt)-nt),1,0],[0,0,np.cos(nt)]])
    Ad_rv=np.array([[np.sin(nt)/n,2*(1-np.cos(nt))/n,0],[2*(np.cos(nt)-1),(4*np.sin(nt)-3*nt)/n,0],[0,0,np.sin(nt)/n]])
    Ad_vr=np.array([[3*n*np.sin(nt),0,0],[6*n*(np.cos(nt)-1),0,0],[0,0,-n*np.sin(nt)]])
    Ad_vv=np.array([[np.cos(nt),2*np.sin(nt),0],[-2*np.sin(nt),4*np.cos(nt)-3,0],[0,0,np.cos(nt)]])

    Ad=np.concatenate((np.concatenate((Ad_rr, Ad_rv), axis=1),
                       np.concatenate((Ad_vr, Ad_vv), axis=1)), axis=0)
    Bc=np.concatenate((np.zeros((3,3)),np.eye(3)))
    Bd=Ad@Bc
    return Ad, Bd

def scipy_integrator(func, t_span, x0, Xv, K, n, atol=1e-12, rtol=1e-12,
                     method='DOP853', events=None):
    t0, tf = t_span[0], t_span[-1]
    Xref_sol = solve_ivp(lambda t,X: func(t,X,Xv,K,n),
                            [t0,tf], x0, t_eval=t_span,
                            atol=atol, rtol=rtol, method = method, events=events)
    tref_out, Xref_out = Xref_sol.t, Xref_sol.y
    return tref_out, Xref_out

class CWHEnv(gym.Env):
    def __init__(self, r, w_pos, w_vel, w_control,tf,n_t_span,param_const):
        super(CWHEnv, self).__init__()
        self.mu     = 398600.4418 # [km3/sec2]
        self.r      = r # [km]
        self.n      = np.sqrt(self.mu/r**3) # [rad/sec]
        A,B=dynamics(self.n )
        Q,R=np.diag([w_pos,w_pos,w_pos,w_vel,w_vel,w_vel]),w_control*np.eye(3)
        K, S, E = lqr(A, B, Q, R)
        self.K = -K
        self.t_span   = np.linspace(0,tf, n_t_span)
        self.Xc = np.zeros(6)
        self.cnt       = 0
        self.param_const = param_const

    def reset(self,s0):
        super().reset()
        self._location = s0 # why '_' is in front of the variable name?
        self.Xv_k      = s0 # initially set it to be the Deputy spacecraft state


        return self._observation(), {}

    def _observation(self):
        return self._location

    def step(self, action):
        # kappa=action # scaling factor
        # Xv_kp1 = self.Xv_k + kappa*(self.Xc-self.Xv_k)
        Xv_kp1=np.zeros(6)
        Xv_kp1[:3]=generate_virtual_target(self.Xc,self._location,action,self.param_const)
        self.Xv_k = Xv_kp1

        s          = self._location
        K, n       = self.K, self.n
        t_span, cnt = self.t_span, self.cnt
        t_span_step = np.array([t_span[cnt], t_span[cnt+1]])
        _, Xref_out=scipy_integrator(eom_cwh, t_span_step, s, Xv_kp1, K, n, atol=1e-12, rtol=1e-12,
                             method='DOP853', events=None)

        self._location = Xref_out[:,-1]
        # reward=reward_func_pred(t_span[cnt],t_span[-1],s,Xv_kp1,K,n,param_const)
        reward=reward_func(t_span[cnt],s,Xv_kp1,K,self.param_const)
        self.cnt+=1
        terminated = 1 if np.linalg.norm(self._location[0:3])<1e-2 else 0
        info = {}
        return self._observation(), reward, terminated, False, info


def random_action():
    return [0.01, 0.01, 0.01]
    # return random.uniform(0,1)

def generate_dummy_action(): # dummy 3D action generater
    action=np.random.uniform(low=0.0,high=1.0,size=3)

    return action

def generate_virtual_target(Xc,Xd,action,param_const): # transform action to the virtual target
    rel_dist=np.linalg.norm(Xd[:3]-Xc[:3]) # distance between the deputy and chief
    r=rel_dist*action[0]
    r_tmp = np.zeros(3)
    if r<=param_const["activate_distance"]: # inside of the vicinity of the chief spacecraft
        # action determines the virtual target within the cone.
        r_tmp[0] = -1
        r_tmp[1], theta = np.tan( np.deg2rad(param_const["alpha_deg"]) )*action[1], (2*np.pi)*action[2]
        r_tmp=r*r_tmp/np.linalg.norm(r_tmp)
        dcm_theta=dcm_1axis(theta,axis=0)
        Xv = dcm_theta@r_tmp+Xc[:3]
    else: # outside of the vicinity of the chief spacecraft
        # action determines the virtual target within a ball with radius of a relative distance.
        r_tmp[0] = -r
        phi, theta = (2*np.pi)*action[1], (2*np.pi)*action[2]
        dcm_phi, dcm_theta   = dcm_1axis(phi,axis=2), dcm_1axis(theta,axis=1)
        Xv = dcm_theta@dcm_phi@r_tmp+Xc[:3]

    return Xv

# def generate_naive_virtual_target(Xc,action,param_const): # transform action to the virtual target
#     r_tmp = np.zeros(3)
#     r_tmp[0], phi, theta = -action[0], (2*np.pi)*action[1], (2*np.pi)*action[2]
#     dcm_phi, dcm_theta   = dcm_1axis(phi,axis=2), dcm_1axis(theta,axis=1)

#     Xv = dcm_theta@dcm_phi@r_tmp+Xc

#     return Xv

def dcm_1axis(ang, axis=0):
    if axis == 0:
        M_ang = np.array([
            [1,            0,           0],
            [0,  np.cos(ang), np.sin(ang)],
            [0, -np.sin(ang), np.cos(ang)]
        ])
    elif axis == 1:
        M_ang = np.array([
            [np.cos(ang), 0, -np.sin(ang)],
            [          0, 1,            0],
            [np.sin(ang), 0,  np.cos(ang)]
        ])
    elif axis == 2:
        M_ang = np.array([
            [ np.cos(ang), np.sin(ang), 0],
            [-np.sin(ang), np.cos(ang), 0],
            [           0,           0, 1]
        ])

    return M_ang

def reward_potential_LoS_h1(s,param_const):
    eps     = 1e-12
    c1, c2  = 1, 1 # coefficient
    rel_pos = s[0:3]
    mag_rel_pos=np.linalg.norm(rel_pos)
    margin_activate=mag_rel_pos-param_const['activate_distance']
    enable_h1=np.where(margin_activate > 0, np.array([1,0]),np.array([0,1]) )

    unit_rel_pos=rel_pos/mag_rel_pos
    R_bar_direction=np.array([-1,0,0])
    cosA=unit_rel_pos.T@R_bar_direction
    HalfConeAng_rad = np.deg2rad(param_const['alpha_deg'])
    tmp_h1=(-cosA+np.cos(HalfConeAng_rad))

    # guide the Deputy spacecraft to the docking port direciton
    sphere_penalty = c1/np.max([margin_activate, eps])*np.max([tmp_h1, 0]) + eps
    cone_penalty   = c2*(1/np.max([-tmp_h1, eps]) - 1/(1-np.cos(HalfConeAng_rad)) ) + eps
    h1             = np.array([sphere_penalty, cone_penalty])

    reward_h1= -enable_h1@h1   # note positive h1 means violation.
    return reward_h1

def reward_potential_approach_velocity_h3(s,param_const):
    eps = 1e-4
    c3, c4 = 1, 1 # coefficient
    h3_margin, gamma2 = param_const['h3_margin'], param_const['h3_gamma2']
    mag_rel_pos, mag_rel_vel  = np.linalg.norm(s[0:3]), np.linalg.norm(s[3:])
    margin_activate=mag_rel_pos-param_const['activate_distance']
    enable_h3=np.where(margin_activate > 0, np.array([1,0]),np.array([0,1]) )
    vel_enter = gamma2*param_const['activate_distance'] + h3_margin
    
    # outside of Keep Out Sphere (KOS)
    relvel_margin = vel_enter - mag_rel_vel
    outside_kos_h3 = c3*np.min([relvel_margin,0])/np.max([margin_activate,eps])
    
    # close proximity operation
    tmp_h3 = mag_rel_vel - gamma2*mag_rel_pos - h3_margin
    inside_kos_h3 = c4*(1/np.max([-tmp_h3, eps]))

    h3=np.array([outside_kos_h3, inside_kos_h3])
    reward_h3= -enable_h3@h3

    return reward_h3

def reward_single_LoS_h1(s,param_const):
    rel_pos=s[0:3]
    mag_rel_pos=np.linalg.norm(rel_pos)
    enable_h1=np.where(-mag_rel_pos+param_const['activate_distance'] < 0, 0, 1)

    unit_rel_pos=rel_pos/mag_rel_pos
    R_bar_direction=np.array([-1,0,0])
    cosA=unit_rel_pos.T@R_bar_direction
    HalfConeAng_rad = np.deg2rad(param_const['alpha_deg'])

    tmp_h1=(-cosA+np.cos(HalfConeAng_rad))
    h1=np.where(tmp_h1 > 0, 1, 0)
    reward_h1= -enable_h1*h1   # note positive h1 means violation.
    return reward_h1

def reward_single_thrust_h2(Xv_kp1,s,K,param_const):
    del_X = s-Xv_kp1.T
    thrust = K@del_X
    mag_thrust=np.linalg.norm(thrust)
    umax=param_const['umax']
    tmp_h2=mag_thrust-umax
    reward_h2=np.where(tmp_h2 > 0, -1, 0)

    return reward_h2

def reward_single_approach_velocity_h3(s,param_const):
    h3_margin, gamma2 = param_const['h3_margin'], param_const['h3_gamma2']
    mag_rel_pos, mag_rel_vel  = np.linalg.norm(s[0:3]), np.linalg.norm(s[3:])
    enable_h3=np.where(-mag_rel_pos+param_const['activate_distance'] < 0, 0, 1)

    tmp_h3 = mag_rel_vel - gamma2*mag_rel_pos - h3_margin
    h3=np.where(tmp_h3 > 0, 1, 0 )
    reward_h3= -enable_h3*h3

    return reward_h3


def reward_LoS_h1(Xref_out,param_const):
    rel_pos=Xref_out[0:3,:]
    mag_rel_pos=np.linalg.norm(rel_pos,axis=0)
    enable_h1=np.where(-mag_rel_pos+param_const['activate_distance'] < 0, 0, 1)
    unit_rel_pos=rel_pos/mag_rel_pos
    R_bar_direction=np.array([-1,0,0])
    cosA=unit_rel_pos[:,:].T@R_bar_direction
    HalfConeAng_rad = np.deg2rad(param_const['alpha_deg'])
    tmp_h1_hist=(-cosA+np.cos(HalfConeAng_rad))
    h1_hist=np.where(tmp_h1_hist > 0, 1, 0)
    reward_h1= -enable_h1.T@h1_hist   # note positive h1 means violation.

    return reward_h1

def reward_thrust_h2(Xv_kp1,Xref_out,K,param_const):
    tiled_Xv_kp1=np.tile(Xv_kp1.T,(np.shape(Xref_out)[1],1))
    tiled_dX = Xref_out-tiled_Xv_kp1.T
    tiled_thrust = K@tiled_dX
    mag_thrust_hist=np.linalg.norm(tiled_thrust,axis=0) # column norm
    umax=param_const['umax']
    tmp_h2_hist=mag_thrust_hist-umax
    h2_thrust_hist=tmp_h2_hist/np.abs(tmp_h2_hist)
    reward_h2=np.sum(-np.fromiter(map(relu_function,h2_thrust_hist), dtype=np.float64))

    return reward_h2

def reward_approach_velocity_h3(Xref_out,param_const):
    h3_margin, gamma2 = param_const['h3_margin'], param_const['h3_gamma2']
    mag_rel_pos, mag_rel_vel  = np.linalg.norm(Xref_out[0:3,:],axis=0), np.linalg.norm(Xref_out[3:,:],axis=0)
    enable_h3=np.where(-mag_rel_pos+param_const['activate_distance'] < 0, 0, 1)

    tmp_h3_hist = mag_rel_vel - gamma2*mag_rel_pos - h3_margin;
    h3_hist=np.where(tmp_h3_hist > 0, 1, 0 )
    reward_h3= -enable_h3.T@h3_hist
    return reward_h3

def reward_terminal_stability_h4(Xref_out,Xv_kp1,param_const):
    mag_pos=np.linalg.norm((Xref_out[0:3,-1].T-Xv_kp1[0:3]).T) # column norm
    mag_vel=np.linalg.norm((Xref_out[3:,-1].T-Xv_kp1[3:]).T) # column norm

    reward_pos = check_terminal_stability_h4(mag_pos, param_const['pos_terminal_stability']) # docking constraint
    reward_vel = check_terminal_stability_h4(mag_vel, param_const['vel_terminal_stability']) # soft docking constraint
    reward_h4=reward_pos+reward_vel
    return reward_h4

def check_terminal_stability_h4(x, eps):
    return np.where(x < eps, 1, -1)

def reward_convergence_reference_command(Xv_kp1):
    reward_convergence_rg=-np.linalg.norm(Xv_kp1)
    return reward_convergence_rg

# def reward_func_pred(t,tf,s,Xv_kp1,K,n,param_const):
#     # Ad, Bd=DT_dynamics(n,dt) # in future work, MPC can be added.

#     N=10001
#     t_span=np.linspace(t,t+tf,N) # prediction horizon
#     dt=t_span[1]-t_span[0]
#     _, Xref_out=scipy_integrator(eom_cwh, t_span, s, Xv_kp1, K, n, atol=1e-12, rtol=1e-12,
#                          method='DOP853', events=None)

#     # accumulated penalties
#     reward_h1=reward_LoS_h1(Xref_out,param_const)
#     reward_h2=reward_thrust_h2(Xv_kp1,Xref_out,K,param_const)
#     reward_h3=reward_approach_velocity_h3(Xref_out,param_const)
#     reward_h4=reward_terminal_stability_h4(Xref_out,Xv_kp1,param_const)
#     reward_convergence_rg=reward_convergence_reference_command(Xv_kp1)

#     # reward_terminate (s) ; function of distance ? or delta function wise reward?
#     # time penalty +


#     r = reward_h1 + reward_h2 + reward_h3 + N*reward_h4 + 1e2*reward_convergence_rg
#     return r

def reward_func(t,s,Xv_kp1,K,param_const):
    reward_pos=100
    reward_vel=1
    scaling_time_penalty=reward_pos**2

    ## penalties from potential functions
    penalty_h1=reward_potential_LoS_h1(s,param_const)

    ## penalties at the current time instant
    # penalty_h1=reward_single_LoS_h1(s,param_const)
    penalty_h2=reward_single_thrust_h2(Xv_kp1,s,K,param_const)
    penalty_h3=reward_single_approach_velocity_h3(s,param_const)

    ## We want to bring Xv_kp1 close enough to the origin (Chief spacecraft)
    reward_Xv=reward_pos*(1e-2/np.linalg.norm(Xv_kp1[0:3])) + reward_vel*(1e-5/np.linalg.norm(Xv_kp1[3:]))
    #reward_Xv_vel=reward_vel*(1e-5/np.linalg.norm(Xv_kp1[3:]))
    # reward_Xv = 0

    ## terminate condition
    # reward_terminate (s) ; function of distance ? or delta function wise reward?
    reward_terminate=np.where(np.linalg.norm(s[0:3])<1e-2,reward_pos**2,0)

    # time penalty - 10 hrs limit
    penalty_time=np.where(t>3600*10,-scaling_time_penalty,0)
    penalty_time = 0

    # r = reward_h1 + reward_h2 + reward_h3 + N*reward_h4 + 1e2*reward_convergence_rg # In the future, additional constraints can be considered.
    r = penalty_h1 + penalty_h2 + penalty_h3 + reward_Xv + reward_terminate + penalty_time

    return r

def relu_function(x):
  return np.where(x <= 0, 0, x)



# def reward_thrust(X,Xv,K,param_const):
#     umax = param_const['umax']
#     u=K@(X-Xv)
#     h2=u-umax
#     reward_h2 = -np.where(h2 <= 0, 0, h2)
#     return reward_h2

if __name__ == '__main__':
    param_const = {
        # 'umax': 5e-4,  # [km/s^2], umax
        'umax': 5e2,  # [km/s^2], umax
        'alpha_deg': 20,  # [deg], half-cone angle
        'h1_margin_deg': 2,  # [deg], margin for half-cone angle
        'h3_margin': 1e-4,
        'h3_gamma2': 5,  # soft-docking parameters
        'activate_distance': 1,  # [km], distance to activate soft-docking constraint
        'pos_terminal_stability': 1e-2, # [km], (<10 m) terminal stability for position
        'vel_terminal_stability': 1e-5 # [km/sec], (<1 cm/sec)terminal stability for velocity
    }


    plt.close('all')
    R_earth  = 6378.137 # [km], radius of the Earth
    altitude = 550 # [km]
    R = altitude + R_earth
    w_pos, w_vel, w_control = 1e3, 1, 1
    s0 = np.array([-10, 20, 20, 0.5, 0.3, 0.1]); # initial state
    tf = 10 # [sec], final simulation time
    n_t_span=1001 # number of simulation time steps

    env = CWHEnv(R,w_pos,w_vel,w_control, tf, n_t_span, param_const)
    obs, info = env.reset(s0)
    # env.render()
    print('initial state: {}'.format(obs))
    for step in range(1000):
        #time.sleep(0.5)
        next_state, reward, terminated, truncated, info = env.step(generate_dummy_action())
        # env.render()
        print('next state: {}. reward: {}'.format(next_state, reward))
        if terminated:
            break