import time
import torch
import torch.distributed as dist
import numpy as np
from torchdiffeq import odeint
from dataclasses import dataclass
from enum import Enum

from .adaptivity import _adaptively_add_y, _find_excess_y, _find_sparse_y, _compute_error_ratios


class steps(Enum):
    FIXED = 0
    ADAPTIVE = 1

class degree(Enum):
    P = 0
    P1 = 1

@dataclass
class IntegralOutput():
    integral: torch.Tensor
    t: torch.Tensor
    h: torch.Tensor
    y: torch.Tensor
    errors: torch.Tensor
    error_ratios: torch.Tensor
    remove_mask: torch.Tensor
    

class SolverBase():
    def __init__(self, atol, rtol, ode_fxn=None, t_init=0., t_final=1.) -> None:

        self.atol = atol
        self.rtol = rtol
        self.ode_fxn = ode_fxn
        self.t_init = t_init
        self.t_final = t_final

    def _calculate_integral(self, t, y, y0=0, degr=degree.P1):
        raise NotImplementedError
    
    def integrate(self, ode_fxn, y0=0., t_init=0., t_final=1., t=None, ode_args=None):
        raise NotImplementedError

    def _error_norm(self, error):
        return torch.sqrt(torch.mean(error**2, -1))



class SerialAdaptiveStepsizeSolver(SolverBase):
    def __init__(self, solver, atol, rtol, ode_fxn=None, t_init=0, t_final=1.) -> None:
        super().__init__(
            atol=atol,
            rtol=rtol,
            ode_fxn=ode_fxn,
            t_init=t_init,
            t_final=t_final
        )

        self.solver = solver

    
    def integrate(self, ode_fxn=None, y0=torch.tensor([0], dtype=torch.float), t_init=0., t_final=1., t=None, ode_args=None):
        ode_fxn = self.ode_fxn if ode_fxn is None else ode_fxn
        assert ode_fxn is not None, "Must specify ode_fxn or pass it during class initialization."
        assert len(ode_fxn(torch.tensor([[t_init]])).shape) >= 2
        if t is None:
            t=torch.tensor([t_init, t_final])
        
        integral = odeint(
            func=ode_fxn,
            y0=y0,
            t=t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )

        return IntegralOutput(
            integral=integral[-1],
            t=t,
            h=None,
            y=None,
            errors=None,
            error_ratios=None,
            remove_mask=None
        )



class ParallelAdaptiveStepsizeSolver(SolverBase):
    def __init__(self, p, atol, rtol, remove_cut=0.1, ode_fxn=None, t_init=0, t_final=1.):
        super().__init__(
            atol=atol,
            rtol=rtol,
            ode_fxn=ode_fxn,
            t_init=t_init,
            t_final=t_final
        )

        assert p > 0, 'The order of the method must be positive and > 0'
        self.p = p
        self.atol = atol
        self.rtol = rtol
        self.remove_cut = remove_cut
        self.previous_t = None
        self.previous_ode_fxn = None

    def integrate(self, ode_fxn=None, state=None, y0=0., t_init=0., t_final=1., t=None, ode_args=None, remove_mask=None):
        verbose=False
        ode_fxn = self.ode_fxn if ode_fxn is None else ode_fxn
        assert ode_fxn is not None, "Must specify ode_fxn or pass it during class initialization."
        assert len(ode_fxn(torch.tensor([[t_init]])).shape) >= 2
        if state is not None:
            t = state.times
            remove_mask = state.remove_mask
        if remove_mask is not None:
            t = t[~remove_mask]
        
        
        t_add = t
        if t_add is None:
            if self.previous_t is None\
                or self.previous_ode_fxn != ode_fxn.__name__:
                t_add = torch.unsqueeze(
                    torch.linspace(t_init, t_final, 7), 1
                )
            else:
                mask = (self.previous_t[:,0] <= t_final)\
                    + (self.previous_t[:,0] >= t_init)
                t_add = self.previous_t[mask]

        y_pruned, t_pruned = None, None
        idxs_add = torch.arange(len(t_add))
        while len(t_add) > 0:
            if verbose:
                print("BEGINNING LOOP")
                print("TADD", t_add.shape, t_add, idxs_add)
            # Evaluate new points and add new evals and points to arrays
            y, t = _adaptively_add_y(
                ode_fxn, y_pruned, t_pruned, t_add, idxs_add
            )
            if verbose:
                print("NEW T", t.shape, t)
                print("NEW Y", y.shape, y)

            # Evaluate integral
            integral_p, y_p, _ = self._calculate_integral(t, y, y0=y0, degr=degree.P)
            integral_p1, y_p1, h = self._calculate_integral(t, y, y0=y0, degr=degree.P1)
            
            # Calculate error
            error_ratios, error_ratios_2steps = _compute_error_ratios(
                y_p, y_p1, self.rtol, self.atol, self._error_norm
            )
            assert (len(y) - 1)//self.p == len(error_ratios)
            assert (len(y) - 1)//self.p - 1 == len(error_ratios_2steps)
            #print(error_ratios)
            if verbose:
                print("ERROR1", error_ratios)
                print("ERROR2", error_ratios_2steps)
                print(integral_p, integral_p1)
            
            # Create mask for remove points that are too close
            remove_mask = _find_excess_y(self.p, error_ratios_2steps, self.remove_cut)
            assert (len(remove_mask) == len(t))
            if verbose:
                print("RCF", remove_mask)

            y_pruned = y[~remove_mask]
            t_pruned = t[~remove_mask]

            # Find indices where error is too large to add new points
            # Evaluate integral
            _, y_p_pruned, _ = self._calculate_integral(
                t_pruned, y_pruned, y0=y0, degr=degree.P
            )
            _, y_p1_pruned, _ = self._calculate_integral(
                t_pruned, y_pruned, y0=y0, degr=degree.P1
            )
            
            # Calculate error
            error_ratios_pruned, _ = _compute_error_ratios(
                y_p_pruned, y_p1_pruned, self.rtol, self.atol, self._error_norm
            )

            t_add, idxs_add = _find_sparse_y(
                t_pruned, self.p, error_ratios_pruned
            )

        self.previous_ode_fxn = ode_fxn.__name__
        self.t_previous = t
        return IntegralOutput(
            integral=integral_p1,
            t=t,
            h=h,
            y=y_p1,
            errors=torch.abs(y_p - y_p1),
            error_ratios=error_ratios,
            remove_mask=remove_mask
        )
