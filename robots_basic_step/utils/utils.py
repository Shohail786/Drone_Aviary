import time

def sync(i,start_time,timestep):
    """
    Syncs the stepped simulation with the wall clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    :param i: int
        Current simulation iteration.
    :param start_time: timestamp
        Timestap of the simulation start.
    :param timestep: float
        Desired, wall-clock step of the simulation's rendering
    :return:
    """
    if timestep>0.04 or i%(int(1/(24*timestep)))==0:
        elapsed=time.time()-start_time
        if elapsed<(i*timestep):
            time.sleep(timestep*i-elapsed)