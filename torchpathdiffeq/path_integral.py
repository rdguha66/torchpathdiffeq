from .solvers import SerialAdaptiveStepsizeSolver
from .runge_kutta import RKParallelAdaptiveStepsizeSolver



def ode_path_integral(ode_fxn, y0, t, solver, atol, rtol, computation='parallel', remove_cut=0.1, t_init=0, t_final=1):

    if computation.lower() == 'parallel':
        integrator = RKParallelAdaptiveStepsizeSolver(
            solver=solver,
            ode_fxn=ode_fxn,
            atol=atol,
            rtol=rtol,
            remove_cut=remove_cut,
            t_init=t_init,
            t_final=t_final
        )
    elif computation.lower() == 'serial':
        integrator = SerialAdaptiveStepsizeSolver(
            solver=solver,
            atol=atol,
            rtol=rtol,
            ode_fxn=ode_fxn,
            t_init=t_init,
            t_final=t_final
        )
    else:
        raise ValueError(f"Path integral computation type must be 'parallel' or 'serial', not {computation}.")

    integral_output = integrator.integrate(
        t_init=t_init,
        t_final=t_final
    )

    return integral_output