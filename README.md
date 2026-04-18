================================================================================
MR DAMPER vs PASSIVE DAMPER — ROCKET LANDING SIMULATION
================================================================================
WHAT THIS STUDY IS:
    A computational simulation comparing two types of landing leg dampers:
      1. PASSIVE DAMPER  — fixed damping coefficient, cannot adapt
      2. MR DAMPER       — variable damping coefficient, adapts every 1ms


WHAT A DAMPER DOES:
    During landing, the rocket hits the ground with kinetic energy = 0.5*m*v^2.
    The landing leg spring and damper must absorb this energy before it reaches
    the rocket frame. The damper dissipates energy as heat through fluid
    resistance. The spring stores and returns energy elastically.
    We want to MINIMIZE the peak force the rocket frame experiences.


PHYSICS MODEL:
    The landing leg is a 1-DOF mass-spring-damper:
        m * x'' = -k*x - c*x'
    where:
        m   = rocket mass (kg)
        x   = leg compression (m), positive = compressed
        x'  = velocity of compression (m/s)
        x'' = acceleration (m/s^2)
        k   = spring stiffness (N/m)
        c   = damping coefficient (N·s/m)


KEY METRIC — WHY WE USE PEAK SPRING FORCE:
    The total force at any instant = k*x + c*x'
    At t=0 (first contact), x=0 but velocity is maximum, so c*x' dominates.
    For a high c_max, this creates a huge spike at t=0 that is a model artifact
   
    Instead we measure PEAK SPRING FORCE = max(k*x) over the full simulation.
    This is the force at MAXIMUM LEG COMPRESSION, the true structural load
    the rocket frame sustains. It is physically meaningful and avoids the
    t=0 artifact entirely.


NUMERICAL METHOD — RK4:
    Because c changes every timestep (MR damper), we cannot solve analytically.
    We step forward through time using Runge-Kutta 4th Order (RK4) integration.
    RK4 computes 4 slope estimates per step and takes a weighted average,
    giving 4th-order accuracy (error ~ dt^4, negligible at dt=0.001s).


CONTROL LAW — SKYHOOK ALGORITHM:
    Named after the concept of damping against a fixed point in the sky.
    Rule: apply HIGH damping when compressing, LOW damping when rebounding.
    This ensures we only dissipate energy, never add it at the wrong moment.
    Reference: Karnopp, Crosby, Harwood (1974), J. Engineering for Industry.


AI tools were used in assistance to write this program
================================================================================
