def calculateSpeedup(sequential_time, parallel_time):

    return sequential_time / parallel_time

def calculateEfficiency(speedup, NUM_PROCESSORS):

    return speedup / NUM_PROCESSORS

def calculateAmdhalUpperLimit(P):
    return 1 / (1-P)
def calculateAmdhal(P,NUM_PROCESSORS):
    return 1 / ((1-P) + (P / NUM_PROCESSORS))

def calculateGustafson(P,NUM_PROCESSORS):
    a = 1-P
    return NUM_PROCESSORS + a*(1-P)

def interpretSpeedup(speedup, NUM_PROCESSORS):
    interpretation = ""
    if speedup == NUM_PROCESSORS:
        interpretation += f"Speedup: {speedup:4f} : NP: {NUM_PROCESSORS} : Linear Speedup. The Task perfectly scales with Number of Processors"
    elif speedup < NUM_PROCESSORS:
        interpretation += f"Speedup: {speedup:4f} : NP: {NUM_PROCESSORS} : Real Speedup."
    elif speedup > NUM_PROCESSORS:
        interpretation += f"Speedup: {speedup:4f} : NP: {NUM_PROCESSORS} : Superlinear Speedup."
    print(interpretation)
def interpretEfficiency(efficiency, NUM_PROCESSORS):
    interpretation = ""
    print(efficiency)
    if efficiency == 1:
        interpretation += f"Efficiency: {efficiency:4f} : NP: {NUM_PROCESSORS} : Linear Case. Each Processor Contributes Equally without any overhead"
    elif efficiency < 1 and efficiency > 0.5:
        interpretation += f"Efficiency: {efficiency:4f} : NP: {NUM_PROCESSORS} : Real Case. Efficiency Reduced due to overhead such as communication or synchronization"
    elif efficiency < 0.5:
        interpretation += f"Efficiency: {efficiency:4f} : NP: {NUM_PROCESSORS} : Low Efficiency. Efficiency Low due to high communication and synchronization overhead,indicating the task is not optimally parallelizable\n"
    else:
        interpretation += f"Efficiency: {efficiency:4f} : NP: {NUM_PROCESSORS}"
    print(interpretation)

def interpretAmdahl(P, upper_limit, amdahl):
    a = 1-P
    interpretation = f"With Parallelizable Portion of {P:4f}, the presence of a serial component {a:4f} sets a definitive upper bound on achievable speedup to {upper_limit:4f}, even when increasing processor count\n"
    interpretation += f"The current speedup is {amdahl:4f}.\n"
    print(interpretation)

def interpretGustafson(P, gustafson):
    a = 1-P
    interpretation = f"The achievable speedup is {gustafson:4f}. By scaling the problem size with number of processors, the system behaves as if it is {gustafson:4f} times faster than the sequential version, despite the presence of sequential portion of {a:4f}\n"
    print(interpretation)

def interpretMetrics(NUM_PROCESSORS, speedup, efficiency, P, amdahl_upper_limit, amdahl, gustafson):
    interpretSpeedup(speedup, NUM_PROCESSORS)
    interpretEfficiency(efficiency, NUM_PROCESSORS)
    interpretAmdahl(P, amdahl_upper_limit, amdahl)
    interpretGustafson(P, gustafson)

    
    

        