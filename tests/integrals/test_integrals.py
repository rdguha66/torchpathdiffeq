import torch
import pytest

from torchpathdiffeq.runge_kutta import RKParallelAdaptiveStepsizeSolver

integrator_list = [RKParallelAdaptiveStepsizeSolver]

def identity(t):
    return 1

def identity_solution(t_init, t_final):
    return t_final - t_init

def t(t):
    return t

def t_solution(t_init, t_final):
    return 0.5*(t_final**2 - t_init**2)

def t_squared(t):
    return t**2

def t_squared_solution(t_init, t_final):
    return (t_final**3 - t_init**3)/3.

def sine_squared(t, w=0.2):
    return torch.sin(t*w*2*torch.pi)**2

def sine_squared_solution(t_init, t_final, w=0.2):
    _w = 4*torch.pi*w
    return (t_final - t_init)/2.\
        + (torch.sin(torch.tensor([_w*t_init])) - torch.sin(torch.tensor([_w*t_final])))/(2*_w)

ODE_dict = {
    "t" : (t, t_solution),
    "t_squared" : (t_squared, t_squared_solution),
    "sine_squared" : (sine_squared, sine_squared_solution)
}

def test_integrals():
    cutoff = 0.01
    t_init = 0.
    t_final = 1.
    integrator = RKParallelAdaptiveStepsizeSolver(p=1, atol=0.001, rtol=0.0001, remove_cut=0.105)
    for name, (ode, solution) in ODE_dict.items():
        integral_output = integrator.integrate(ode, t_init=t_init, t_final=t_final)
        correct = solution(t_init=t_init, t_final=t_final)
        error_string = f"Failed to properly integrate {name}, calculated {integral_output.integral} but expected {correct}"

        assert torch.abs(integral_output.integral - correct)/correct < cutoff, error_string