#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DMSC Plots


# In[2]:


import os
os.environ['PYFOAM_GLOBAL_CONFIG'] = r'C:\Users\carmi\.pyFoam'


# In[3]:


from PyFoam.Execution.ConvergenceRunner import ConvergenceRunner
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer
from PyFoam.RunDictionary.SolutionFile import SolutionFile
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory

solver = "rhoCentralFoam"
case = "bluntNoseHypersonic"
pCmd = "calcPressureDifference"
mCmd = "calcMassFlow"

# Setup
dire = SolutionDirectory(case, archive="MachSweep")
dire.clearResults()
dire.addBackup("HypersonicSolve.logfile")
dire.addBackup("HypersonicSolve.analyzed")

sol = SolutionFile(dire.initialDir(), "U")

# Sweep settings
minMach = 5
maxMach = 15
steps = 5
gamma = 1.4
R = 287
T = 300
a = (gamma * R * T) ** 0.5

# Output file
f = dire.makeFile("MachSweepResults")

for i in range(steps + 1):
    M = minMach + i * (maxMach - minMach) / steps
    U = M * a

    print(f"Setting Mach {M:.1f} → Velocity {U:.2f} m/s")

    sol.replaceBoundary("inlet", f"({U} 0 0)")

    run = ConvergenceRunner(BoundingLogAnalyzer(), argv=[solver, ".", case], silent=True)
    run.start()

    print("Last Time = ", dire.getLast())

    # Pressure difference
    pUtil = UtilityRunner(argv=[pCmd, ".", case], silent=True, logname="Pressure")
    pUtil.add("deltaP", "Pressure at .* Difference .*\] (.+)")
    pUtil.start()
    deltaP = pUtil.get("deltaP")[0]

    # Mass flow
    mUtil = UtilityRunner(argv=[mCmd, ".", case, "-latestTime"], silent=True, logname="MassFlow")
    mUtil.add("mass", "Flux at (.+?) .*\] (.+)", idNr=1)
    mUtil.start()
    massFlow = mUtil.get("mass", ID="outlet")[0]

    dire.lastToArchive(f"mach={M:.1f}")
    dire.clearResults()

    print(f"Mach: {M:.1f}, ΔP: {deltaP}, Mass Flow: {massFlow}")
    f.writeLine((M, deltaP, massFlow))

sol.purgeFile()


# In[ ]:





# In[ ]:




