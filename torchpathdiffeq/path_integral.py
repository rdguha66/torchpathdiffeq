from .solvers import SerialAdaptiveStepsizeSolver
from .runge_kutta import RKParallelAdaptiveStepsizeSolver

ADAPTIVE_SOLVER_P = {
    'euler' : 1,
    'heun' : 2
}

def ode_path_integral(ode_fxn, y0, t, method, atol, rtol, remove_cut=0.1, t_init=0, t_final=1):

    assert method.lower() in ADAPTIVE_SOLVER_P

    integrator = RKParallelAdaptiveStepsizeSolver(
        p=ADAPTIVE_SOLVER_P[method],
        ode_fxn=ode_fxn,
        atol=atol,
        rtol=rtol,
        remove_cut=remove_cut,
        t_init=t_init,
        t_final=t_final
    )

    integral_output = integrator.integrate(
        t_init=t_init,
        t_final=t_final
    )

    return integral_output