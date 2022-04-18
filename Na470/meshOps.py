
import os
import os.path as osp
import sys
import time
from capytaine import FloatingBody 
import meshmagick.mesh as mmm
import meshmagick.mmio as mmio
import meshmagick.mesh_clipper as mmc
import meshmagick.hydrostatics as mmhs
import meshmagick.MMviewer as mmv
from scipy.linalg import block_diag
import numpy as np
#from stl import mesh
import hymesh as hm
import logging 
import capytaine as cpt
from capytaine.io.xarray import separate_complex_values

#Making Test Directory
testName = f'hydroData'
resDir = f'./../POTS/{testName}/'
runName = 'test_runs4'
pathToMesh = os.path.join(resDir, testName)
if not os.path.exists(resDir):
    os.makedirs(resDir)


logging.basicConfig(level=logging.INFO,
                  format ="%(levelname)s:\t%(message)s")
                  
#Loading Mesh, translating to correct location, adding DOF's
#POTS_mesh = mesh.Mesh.from_file('POTS_Platform_Meshed_Up_Clipped.stl')

#POTS_mesh = FloatingBody(POTS_mesh)
#POTS_mesh = FloatingBody.from_file('POTS_Platform_Meshed_Up_Clipped.stl', file_format='STL')
filePath = f'./clipped_mesh_files/Clipped_FourRoundLegs_1968_Tris.stl'
POTS_mesh = FloatingBody.from_file(filePath)  # gmsh file
POTS_mesh2 = FloatingBody.from_file(filePath)  # gmsh file
POTS_mesh3 = FloatingBody.from_file(filePath)
POTS_mesh4 = FloatingBody.from_file(filePath)


#POTS_mesh.center_of_mass(32.5,32.5,8.25)
#POTS_mesh.translate_z(3)
#POTS_mesh2.translate_z(3.249)
POTS_mesh2.translate_x(68)
POTS_mesh3.translate_y(68)
POTS_mesh4.translate_x(68)
POTS_mesh4.translate_y(68)
#POTS_mesh.show()
POTS_mesh.add_all_rigid_body_dofs()  #Adding all DOF for body 
POTS_mesh2.add_all_rigid_body_dofs()  #Adding all DOF for body 
POTS_mesh3.add_all_rigid_body_dofs()  #Adding all DOF for body 
POTS_mesh4.add_all_rigid_body_dofs()  #Adding all DOF for body 


bodies = POTS_mesh + POTS_mesh2 + POTS_mesh3 + POTS_mesh4
bodies.show()
#POTS_mesh = POTS_mesh.keep_immersed_part(sea_bottom=-100, inplace=False)

#Defining problems, frequency range, depth etc
omega_range = np.linspace(0.2, 15.0, 20)

#Setting up added mass problem 
problems = [ 
  cpt.RadiationProblem(body = bodies, rho =1025.0, radiating_dof= dof, omega = omega,free_surface =0.0, sea_bottom = -100.0)
  for dof in bodies.dofs
  for omega in omega_range
]

#Setting Up rasdiation damping problem

problems += [
  cpt.DiffractionProblem(body=bodies,rho=1025.0,omega= omega, free_surface = 0.0, sea_bottom = -100.0)
  for omega in omega_range
]
#problems = [radiation_problem, diffraction_problem]
# Solving problems
#timeStart =time.time
print(f'running case: {runName}')
direct_linear_solver = cpt.BasicMatrixEngine(linear_solver = 'direct')
solver = cpt.BEMSolver(engine=direct_linear_solver)
results = [solver.solve(pb) for pb in sorted(problems)]

data = cpt.assemble_dataset(results)
separate_complex_values(data).to_netcdf(osp.join(resDir,f'{runName}.nc'),
                                        encoding={'radiating_dof':{'dtype':'U'},
                                        'influenced_dof':{'dtype':'U'}})

#timeEnd = time.time()
#timeComp = timeEnd-timeStart
#timeLog = open(osp.join(resDir, f'comp_time.seconds'), 'a')
#timeLog.write(f'{timeComp:.3f}\n')
#timeLog.close()
import matplotlib.pyplot as plt

plt.figure()
for dof in bodies.dofs:
  plt.plot(
    omega_range,
    data['added_mass'].sel(radiating_dof = dof, influenced_dof = dof),
    label = dof,
    marker = 'o',
  )
plt.xlabel('omega')
plt.ylabel('added_mass')
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('./results/Added_Mass_436.png')

plt.figure()
for dof in bodies.dofs:
  plt.plot(
    omega_range,
    data['radiation_damping'].sel(radiating_dof = dof, influenced_dof = dof),
    label = dof,
    marker = 'o',
  )
plt.xlabel('omega')
plt.ylabel('radiation damping')
plt.legend()
plt.tight_layout()
plt.show()
#plt.savefig('./results/Radiation_Damping_436.png')







