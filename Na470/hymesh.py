import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# some constants
d2r = np.pi/180
r2d = 180/np.pi

# some basic geometry functions (create lines/arcs in 3D space, apply
# cosine spacing, etc)
def create_arc(startAngle, endAngle, center, radius, nPts):
    '''create arc between 2 points - constant spacing'''
    # angular spacing of torus points
    thetaArc = np.linspace(startAngle*d2r, endAngle*d2r, nPts)
    # circle centre for torus
    centerX = center[0] #x_bottom[0] - radius*np.sin(135*d2r)
    centerZ = center[1] #z_bottom[0] - radius*np.cos(135*d2r)
    # compute x and z for torus
    xPts = centerX + radius*np.sin(thetaArc)
    yPts = np.zeros(len(xPts))
    zPts = centerZ + radius*np.cos(thetaArc)
    pts = np.array([xPts, yPts, zPts])
    return pts

def linSpace3D_A2B(ptA, ptB, nPts):
    '''create linear spacing between 2 points in 3D space (x, y, z)'''
    linSpaceX = np.linspace(ptA[0], ptB[0], nPts)
    linSpaceY = np.linspace(ptA[1], ptB[1], nPts)
    linSpaceZ = np.linspace(ptA[2], ptB[2], nPts)
    return np.asarray([linSpaceX, linSpaceY, linSpaceZ])

def create_linSpace(refining, pts):
    '''create linspace for cosine spacing'''
    if refining == 'start':
        minVal = 90
        maxVal = 180
    if refining == 'end':
        minVal = 0
        maxVal = 90
    if refining == 'both':
        minVal = 0
        maxVal = 180
    return np.linspace(minVal, maxVal, pts)

def create_cosSpace(length, refining, pts):
    '''convert linSpace to cosSpace for desired length'''
    linSpace = create_linSpace(refining, pts)
    cosLin = np.cos(d2r*linSpace)
    minCosLin = min(cosLin)
    if refining == 'both':
        length *= 0.5
    cosSpace = (cosLin+abs(minCosLin))*length
    return np.sort(cosSpace)

def linSpace_to_cosSpace(linSpace3D, refining='both'):
    maxValX = max(linSpace3D[0,:])
    minValX = min(linSpace3D[0,:])
    lenValX = maxValX - minValX
    lenX = len(linSpace3D[0,:])

    maxValY = max(linSpace3D[1,:])
    minValY = min(linSpace3D[1,:])
    lenValY = maxValY - minValY
    lenY = len(linSpace3D[1,:])

    maxValZ = max(linSpace3D[2,:])
    minValZ = min(linSpace3D[2,:])
    lenValZ = maxValZ - minValZ
    lenZ = len(linSpace3D[2,:])

    minVals = [minValX, minValY, minValZ]

    if refining == 'both':
        minCos = 0
        maxCos = 180
    if refining == 'start':
        minCos = 180
        maxCos = 90
    if refining == 'end':
        minCos = 90
        maxCos = 180

    lenCos = maxCos - minCos
    degStep = lenCos/lenX #TODO: generalize
    cosXAxis = np.arange(minCos, (maxCos+degStep), degStep)
    cosYAxis = np.cos(d2r*cosXAxis)
    cosSpacingX = cosYAxis*lenValX
    cosSpacingY = cosYAxis*lenValY
    cosSpacingZ = cosYAxis*lenValZ
    cosSpacing = np.asarray([cosSpacingX, cosSpacingY, cosSpacingZ])
    if refining == 'both':
        for i in [0, 1, 2]:
            cosSpacing[i,:] += abs(min(cosSpacing[i,:]))
            cosSpacing[i,:] /= 2
            cosSpacing[i,:] += minVals[i]
    cosSpacing[0,:] = cosSpacing[0,:][::-1]
    return np.asarray(cosSpacing).T

def create_cosSpace(length, refining, pts):
    '''convert linSpace to cosSpace for desired length'''
    linSpace = create_linSpace(refining, pts)
    cosLin = np.cos(d2r*linSpace)
    minCosLin = min(cosLin)
    if refining == 'both':
        length *= 0.5
    cosSpace = (cosLin+abs(minCosLin))*length
    return np.sort(cosSpace)


# plot profile
def plot_profile(x, z):
    '''plot the profile to be revolved around z axis (x, z coordinates)'''
    plt.scatter(x, z)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# some functions for defining 3D geometries
# axisymmetric meshes
def revolve_profile(x, z, nTheta, revolvingAngle=360.0):
    '''take x, z coordinates and revolve  around z axis at 360/nTheta
    spacing'''

    xLen = len(x)

    # points
    iPoint = 0
    theta = np.linspace(0, revolvingAngle*d2r, nTheta)

    x3d = np.zeros(xLen*nTheta)
    y3d = np.zeros(xLen*nTheta)
    z3d = np.zeros(xLen*nTheta)

    for i in range(nTheta):
        for j in range(len(x)):
            x3d[iPoint] = x[j]*np.cos(theta[i])
            y3d[iPoint] = x[j]*np.sin(theta[i])
            z3d[iPoint] = z[j]
            iPoint = iPoint+1

    points = np.array([x3d, y3d, z3d])

    # panels
    iPanel = 0
    panels = np.zeros((4, (xLen-1)*(nTheta-1)))

    for i in range(xLen-1):
        for j in range(nTheta-1):
            panels[0, iPanel] = (i+1) + xLen*j
            panels[1, iPanel] = (i+1) + 1 + xLen*j
            panels[2, iPanel] = (i+1) + 1 + xLen*(j+1)
            panels[3, iPanel] = (i+1) + xLen*(j+1)
            iPanel = iPanel+1

    return points, panels

# prismatic (extruded profile) meshes
# create face from some outer profile points
def create_face(profileOutline, layers=4, spacing='lin'):
    '''
    create a face from some profile edge points by scaling them down towards
    origin point (0,0,0)

    Parameters
    ----------
    profileOutline :

    Returns
    -------

    Raises
    ------

    Notes
    -----

    '''

    if spacing=='lin':
        scaleFactors = np.linspace(0,1,layers+1)[1:]
    elif spacing=='cos':
        scaleFactors = create_cosSpace(length=1.0, refining='end', pts=layers)
    facePts = np.zeros(min(profileOutline.shape))
    scaleFactors[0] = 0.0
    for sf in scaleFactors:
        layer = sf*profileOutline
        facePts = np.vstack([layer, facePts])

    panels = []
    for i, xyz in enumerate(facePts):
        # create last panel
        if (i+1) == len(facePts)-1:
            panel = [i+1, (i+2)-lenEdgePts, len(facePts), i+1]
            # print(f'last panel {i}: {panel}')
            panels.append(panel)
            break
        # create panels @ inner layer
        if (i+1)>int(((nLayers-1)/nLayers)*len(facePts)):
            panel = [i+1, i+2, len(facePts), i+1]
            # print(f'inner layer, panel {i}: {panel}')
            panels.append(panel)
            continue
        # create panel @ end of layer
        if (i+1)%lenEdgePts == 0:
            panel = [i+1, (i+2)-lenEdgePts, i+2, (i+1)+lenEdgePts]
            # print(f'panel {i} @ end of layer: {panel}')
            panels.append(panel)
            continue
        # create panel
        if i<(lenEdgePts*(nLayers-1) - 1):
            panel = [i+1, i+2, i+(2+lenEdgePts), i+(1+lenEdgePts)]
            # print(f'panel {i}: {panel}')
            panels.append(panel)
    facePanels = np.asarray(panels)

    return facePts, facePanels

def extrude_points(edgePts, dirExtru, lenExtru, numExtru, spacing='cos'):
    '''assume points defined in Oxz plane'''
    cosSpacing = create_cosSpace(lenExtru, 'start', numExtru)
    pts = np.ones((3, max(edgePts.shape)*(numExtru)))
    for i in range(numExtru):
        startIx = (i)*len(edgePts)
        endIx = (i+1)*len(edgePts)
        #print(i, startIx, endIx)
        pts[0, startIx:endIx] = edgePts[:,0]
        pts[1, startIx:endIx] *= cosSpacing[i]
        pts[2, startIx:endIx] = edgePts[:,2]
    return np.asarray(pts).T[len(edgePts):]

def extrude_panels(extrPts, numEdgePts, numSidePts):
    ''''''
    panels = []
    for i in range(len(extrPts)):#-numEdgePts):
        iExtP = numSidePts + i
        #print(f'iExtP = {iExtP}')
        # create panel @ end of first layer
        if (i+1)%numEdgePts == 0:
            if (i+1)/numEdgePts == 1.0:
                panel = [iExtP+1, (iExtP+2)-numEdgePts, (i+2)-numEdgePts, i+1]
                panels.append(panel)
            else:
                panel = [iExtP+1, (iExtP+2)-numEdgePts, (iExtP+2)-(2*numEdgePts),
                         (iExtP+1)-numEdgePts]
                panels.append(panel)
            continue
        # first layer:
        if (i+1)/numEdgePts < 1.0:
            panel = [iExtP+1, iExtP+2, i+2, i+1]
            panels.append(panel)
        elif (i+1)/numEdgePts > 1.0:
            panel = [iExtP+1, iExtP+2, (iExtP+2)-numEdgePts, (iExtP+1)-numEdgePts]
            panels.append(panel)
    return panels


# def create_side_points(profileOutline, layers=4, spacing='lin'):
#     '''create points on one side of mesh by filling in an outline.
#     N.B. the face center is assumed to be (0,0,0)!'''
#     if spacing=='lin':
#         scaleFactors = np.linspace(0,1,layers+1)[1:]
#     elif spacing=='cos':
#         scaleFactors = create_cosSpace(length=1.0, refining='end', pts=layers)
#     facePts = np.zeros(min(profileOutline.shape))
#     scaleFactors[0] = 0.0
#     for sf in scaleFactors:
#         layer = sf*profileOutline
#         facePts = np.vstack([layer, facePts])
#     return facePts

# def create_side_panels(facePts, lenEdgePts, nLayers):
#     "create panels from side points"
#     panels = []
#     for i, xyz in enumerate(facePts):
#         # create last panel
#         if (i+1) == len(facePts)-1:
#             panel = [i+1, (i+2)-lenEdgePts, len(facePts), i+1]
#             # print(f'last panel {i}: {panel}')
#             panels.append(panel)
#             break
#         # create panels @ inner layer
#         if (i+1)>int(((nLayers-1)/nLayers)*len(facePts)):
#             panel = [i+1, i+2, len(facePts), i+1]
#             # print(f'inner layer, panel {i}: {panel}')
#             panels.append(panel)
#             continue
#         # create panel @ end of layer
#         if (i+1)%lenEdgePts == 0:
#             panel = [i+1, (i+2)-lenEdgePts, i+2, (i+1)+lenEdgePts]
#             # print(f'panel {i} @ end of layer: {panel}')
#             panels.append(panel)
#             continue
#         # create panel
#         if i<(lenEdgePts*(nLayers-1) - 1):
#             panel = [i+1, i+2, i+(2+lenEdgePts), i+(1+lenEdgePts)]
#             # print(f'panel {i}: {panel}')
#             panels.append(panel)
#     return np.asarray(panels)


# 3D plotting functions
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_points_3d(xyzs=None, xs=None, ys=None, zs=None):
    '''plot 3D points'''
    if xyzs is not None:
        if len(xyzs[:,0])>len(xyzs[0,:]):
            xyzs = xyzs.T
        xs = xyzs[0,:]
        ys = xyzs[1,:]
        zs = xyzs[2,:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    set_axes_equal(ax)
    plt.show()


# some misc things - maybe better in class
def scale_mesh_pts(pts, sf):
    '''scale all mesh points'''
    return sf*pts

def remove_duplicates(pts):
    '''remove duplicate points'''
    _, ixUniq = np.unique(pts, axis=0, return_index=True)
    ptsUniq = pts[np.sort(ixUniq)]
    return ptsUniq

def translate_y(pts, ty):
    '''translate points in y direction'''
    for pt in pts:
        pt[1]+=ty
    return pts


class Mesh:
    def __init__(self, meshName, points, panels, cog=[0.0, 0.0, 0.0]):
        self.meshName = meshName
        self.points = points
        self.x3d = self.points[0,:]
        self.y3d = self.points[1,:]
        self.z3d = self.points[2,:]
        self.panels = panels
        self.nPts = len(self.points[0,:])
        self.nPanels = np.max(panels.shape)
        self.nemohSaveName = f'{meshName}.nemoh'

    def set_axes_equal(self, ax):
        '''Make axes of 3D plot have equal scale'''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def plot_points_3d(self):
        '''plot 3D points'''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x3d, self.y3d, self.z3d)
        self.set_axes_equal(ax)
        plt.show()

    def save_to_nemoh_mesh(self, saveName=None, XZSym=0):
        '''save mesh points and panels to nemoh format'''
        if saveName==None:
            saveName = self.nemohSaveName
        f = open(saveName, "w")
        f.write(f'2 {int(XZSym)}\n')
        for i in range(self.nPts):
            f.write(f'{i+1} {self.x3d[i]} {self.y3d[i]} {self.z3d[i]}\n')
        f.write('0 0 0 0\n')
        for i in range(self.nPanels):
            f.write(f'{self.panels[0, i]:.0f} {self.panels[1, i]:.0f} {self.panels[2, i]:.0f} {self.panels[3, i]:.0f}\n')
        f.write('0 0 0 0\n')
        f.close()


class Mesh:
    def __init__(self, meshName, points, panels, cog=[0.0, 0.0, 0.0]):
        self.meshName = meshName
        self.points = points
        self.x3d = self.points[0,:]
        self.y3d = self.points[1,:]
        self.z3d = self.points[2,:]
        self.panels = panels
        self.numPoints = len(self.points[0,:])
        self.numPanels = np.max(panels.shape)
        self.nemohSaveName = f'{meshName}.nemoh'
        self.wamitSaveName = f'{meshName}.gdf'
        self.cog = np.asarray(cog)
        self.xBody = np.append(self.cog, 0.0)

    def shear_x(self, xShearFactor):
        shearMat = np.array([[1,0,0], [0,1,0], [-xShearFactor,0,1]])
        for i in range(self.numPoints):
            vec = self.points[:,i]
            self.points[:,i] = vec.dot(shearMat)
        self.cog = self.cog.dot(shearMat)

    def scale_z(self, zScaleFactor):
        self.points[2,:] *= zScaleFactor
        self.cog[2] *= zScaleFactor

    def scale(self, scaleFactor):
        self.points *= scaleFactor
        self.cog *= scaleFactor

    def translate_x(self, xTranslateFactor):
        self.points[0,:] += xTranslateFactor
        self.cog[0] += xTranslateFactor
        cog = self.cog[0]
        return cog

    def translate_y(self, yTranslateFactor):
        self.points[1,:] += yTranslateFactor
        self.cog[1] += yTranslateFactor

    def translate_z(self, zTranslateFactor):
        self.points[2,:] += zTranslateFactor
        self.cog[2] += zTranslateFactor

    def set_axes_equal(self, ax):
        '''Make axes of 3D plot have equal scale'''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def plot_points_3d(self):
        '''plot 3D points'''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x3d, self.y3d, self.z3d)
        self.set_axes_equal(ax)
        plt.show()

    def save_to_nemoh_mesh(self, saveName=None):
        '''save mesh points and panels to nemoh format'''
        if saveName==None:
            saveName = self.nemohSaveName
        f = open(saveName, "w")
        f.write(f'{len(self.panels[0,:])} {len(self.x3d)}\n')
        for i in range(self.numPoints):
            f.write(f'{i+1} {self.x3d[i]} {self.y3d[i]} {self.z3d[i]}\n')
        f.write('0 0 0 0\n')
        for i in range(self.numPanels):
            f.write(f'{self.panels[0, i]:.0f} {self.panels[1, i]:.0f} {self.panels[2, i]:.0f} {self.panels[3, i]:.0f}\n')
        f.write('0 0 0 0\n')
        f.close()

    def save_to_gdf_mesh(self, saveName=None, scale=1.0, gravity=9.81,
                         symmetry=[0, 0]):
        '''save mesh to gdf format'''
        if saveName==None:
            saveName = self.wamitSaveName

        currentTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        f = open(saveName, "w")
        f.write(f'{self.meshName} gdf mesh created at {currentTime} \n')
        f.write(f'{scale}    {gravity}\n')
        f.write(f'{symmetry[0]}    {symmetry[1]}\n')
        f.write(f'{self.numPanels}\n')

        for panel in range(self.numPanels):
            panelNodes = self.panels[:,panel]
            for i in range(4):
                idx = int(panelNodes[i]) - 1
                f.write(f'{self.x3d[idx]:>20.12e} {self.y3d[idx]:>20.12e} {self.z3d[idx]:>20.12e}\n')
        f.close()

    def write_wamit_pot(self, potFileName, meshFileNames, nBodys, xBodys,
                        waterDepth=-1, iRad=1, iDiff=1, nPer=-402,
                        per=[-0.02, 0.02], nBetas=1, betas=[0.0]):
        '''write wamit .pot file'''
        f = open(potFileName, 'w')
        f.write(f'{potFileName}\n')
        f.write(f' {waterDepth:<32} HBOT\n')
        f.write(f' {iRad:<8}{iDiff:<24} IRAD, IDIFF\n')
        f.write(f' {nPer:<32} NPER\n')
        f.write(f' {per[0]:<8}{per[1]:<24} PER\n')
        f.write(f' {nBeta:<32} NBETA\n')
        for beta in betas:
            f.write(f' {beta:<8}')
        f.write(f'BETA\n')
        f.write(f' {nBodys:<32} NBODY\n')
        for i, mesh in enumerate(meshFileNames):
            f.write(f' {mesh:<32}\n')
            f.write(f' {xBodys[i,0]:<10.4f}{xBodys[i,1]:<10.4f}{xBodys[i,2]:<10.4f}{xBodys[i,3]:<10.4f} XBODY(1-4)\n')
            f.write(f' {"1 1 1 1 1 1":<32} IMODE(1-6)\n')
        f.close()

    def write_wamit_fnames(self, modelName, meshFileNames):
        '''write wamit fnames.wam file'''
        f = open(f'fnames.wam', 'w')#, encoding='ascii')
        f.write(f'{modelName}.cfg\n')
        for mesh in meshFileNames:
            f.write(f'{mesh}\n')
        f.write(f'{modelName}.pot\n')
        f.write(f'{modelName}.frc\n')
        f.close()

    def write_wamit_cfg(self, modelName, iLog=1, iPerIn=2, irr=1, iSolve=1, numHeaders=1):
        '''write wamit .cfg file'''
        f = open(f'{modelName}.cfg', 'w')
        f.write(f'! {modelName}\n')
        f.write(f' ILOG={iLog:<19} (1 - panels on free surface)\n')
        f.write(f' IPERIN={iPerIn:<17} (1 - T, 2 - w)\n')
        f.write(f' IRR={irr:<20} (0 - not remove irr freq, 1 - remove irr freq, pannels on free surface)\n')
        f.write(f' ISOLVE={iSolve:<17} (0 - iterative solver, 1 - direct solver)\n')
        f.write(f' NUMHDR={numHeaders:<17} (0 - no output headers, 1 - output headers)\n')
        f.close()

    def write_wamit_frc(self, modelName, meshFileNames, meshCogZ):
        '''write wamit .frc file'''
        f = open(f'{modelName}.frc', 'w')
        f.write(f' {modelName}.frc\n')
        f.write(f' {"1 0 1 0 0 0 0 0 0":<24} IOPTN(1-9)\n')
        for i, mesh in enumerate(meshFileNames):
            f.write(f' {meshCogZ[i]:<24} VCG({i+1})\n')
            f.write(f' {"0.0  0.0  0.0":<24}\n')
            f.write(f' {"0.0  0.0  0.0":<24}\n')
            f.write(f' {"0.0  0.0  0.0":<24} XPRDCT\n')
        f.write(f' {0:<24d} NBETAH\n')
        f.write(f' {0:<24d} NFIELD\n')
        f.close()

    def write_wamit_config(self, ramGBMax=60.0, numCPU=10, licensePath=f'\wamitv7\license'):
        '''write wamit config.wam file'''
        f = open(f'config.wam', 'w')
        f.write(f' generic configuration file:  config.wam\n')
        f.write(f' RAMGBMAX={ramGBMax}\n')
        f.write(f' NCPU={numCPU}\n')
        f.write(f' USERID_PATH={licensePath} \t (directory for *.exe, *.dll, and userid.wam)\n')
        f.write(f' LICENSE_PATH={licensePath}')
        f.close()


class MultiMesh():
    def __init__(self, *Meshes, modelName):
        self.potFileName = f'{modelName}.pot'
        self.cfgFileName = f'{modelName}.cfg'
        self.frcFileName = f'{modelName}.frc'
        self.nBodys = len(Meshes)
        self.Meshes = Meshes
        self.modelName = modelName

    def write_meshes(self, bemCode='wamit'):
        for mesh in self.Meshes:
            if bemCode == 'wamit':
                mesh.save_to_gdf_mesh()
            elif bemCode == 'nemoh':
                mesh.save_to_nemoh_format()

    def write_wamit_pot(self, waterDepth=-1, iRad=1, iDiff=1, nPer=-402,
                        per=[-0.02, 0.02], nBetas=1, betas=[0.0]):
        '''write wamit .pot file'''
        f = open(self.potFileName, 'w')
        f.write(f'{self.potFileName}\n')
        f.write(f' {waterDepth:<32} HBOT\n')
        f.write(f' {iRad:<8}{iDiff:<24} IRAD, IDIFF\n')
        f.write(f' {nPer:<32} NPER\n')
        f.write(f' {per[0]:<8}{per[1]:<24} PER\n')
        f.write(f' {nBetas:<32} NBETA\n')
        for beta in betas:
            f.write(f' {beta:<7}')
        f.write(f'BETA\n')
        f.write(f' {self.nBodys:<32} NBODY\n')
        for mesh in self.Meshes:
            f.write(f' {mesh.wamitSaveName:<32}\n')
            for i in range(4):
                f.write(f' {mesh.xBody[i]:<7}')
            f.write(f'XBODY(1-4)\n')
            f.write(f' {"1 1 1 1 1 1":<32} IMODE(1-6)\n')
        f.close()

    def write_wamit_fnames(self):
        '''write wamit fnames.wam file'''
        f = open(f'fnames.wam', 'w')
        f.write(f'{self.cfgFileName}\n')
        for mesh in self.Meshes:
            f.write(f'{mesh.wamitSaveName}\n')
        f.write(f'{self.potFileName}\n')
        f.write(f'{self.frcFileName}\n')
        f.close()

    def write_wamit_cfg(self, iLog=1, iPerIn=2, irr=1, iSolve=1, numHeaders=1):
        '''write wamit .cfg file'''
        f = open(f'{self.cfgFileName}', 'w')
        f.write(f'! {self.modelName}\n')
        f.write(f' ILOG={iLog:<19} (1 - panels on free surface)\n')
        f.write(f' IPERIN={iPerIn:<17} (1 - T, 2 - w)\n')
        f.write(f' IRR={irr:<20} (0 - not remove irr freq, 1 - remove irr freq, pannels on free surface)\n')
        f.write(f' ISOLVE={iSolve:<17} (0 - iterative solver, 1 - direct solver)\n')
        f.write(f' NUMHDR={numHeaders:<17} (0 - no output headers, 1 - output headers)\n')
        f.close()

    def write_wamit_frc(self):
        '''write wamit .frc file'''
        f = open(f'{self.frcFileName}', 'w')
        f.write(f' {self.frcFileName}\n')
        f.write(f' {"1 0 1 0 0 0 0 0 0":<24} IOPTN(1-9)\n')
        for i, mesh in enumerate(self.Meshes):
            f.write(f' {"0.0":<24} VCG({i+1})\n')
            f.write(f' {"0.0  0.0  0.0":<24}\n')
            f.write(f' {"0.0  0.0  0.0":<24}\n')
            f.write(f' {"0.0  0.0  0.0":<24} XPRDCT\n')
        f.write(f' {0:<24d} NBETAH\n')
        f.write(f' {0:<24d} NFIELD\n')
        f.close()

    def write_wamit_config(self, ramGBMax=32.0, numCPU=6, licensePath=f'\wamitv7\license'):
        '''write wamit config.wam file'''
        f = open(f'config.wam', 'w')
        f.write(f' generic configuration file:  config.wam\n')
        f.write(f' RAMGBMAX={ramGBMax}\n')
        f.write(f' NCPU={numCPU}\n')
        f.write(f' USERID_PATH={licensePath} \t (directory for *.exe, *.dll, and userid.wam)\n')
        f.write(f' LICENSE_PATH={licensePath}')
        f.close()

materials = {'steel' : [7850, 3.0],
             'coated fabric' : [1400, 9.5],
             'sea water' : [1025, 0.0]}

class DiscWithTorus(Mesh):
    def __init__(self, meshName, points, panels,
                 discRadius, discThickness,
                 torusRadiusMinor, torusThickness,
                 discMaterial='steel',
                 torusInnerMaterial='sea water',
                 torusOuterMaterial='coated fabric'):
        super().__init__(meshName, points, panels)
        # disc properties
        self.discRadius = discRadius
        self.discThickness = discThickness
        self.discRho = materials[discMaterial][0]
        self.discVolume = np.pi*discRadius**2 * discThickness
        self.discMass = self.discVolume * self.discRho
        self.discIxx = (1.0/12.0) * self.discMass * (3*self.discRadius**2 +
                                                     self.discThickness**2)
        self.discIyy = self.discIxx
        self.discIzz = (1.0/2.0) * self.discMass * self.discRadius**2

        # torus properties
        self.torusRadiusMinor = torusRadiusMinor
        self.torusThickness = torusThickness
        self.torusRadiusMinorInner = torusRadiusMinor - torusThickness
        self.torusRadiusMajor = torusRadiusMinor + discRadius
        self.torusInnerRho = materials[torusInnerMaterial][0]
        self.torusOuterRho = materials[torusOuterMaterial][0]
        self.torusSurfaceArea = (4 * np.pi**2 * self.torusRadiusMajor *
                                 self.torusRadiusMinor) 
        self.torusOuterVolume = (self.torus_volume(self.torusRadiusMajor,
                                                   self.torusRadiusMinor)
                                 - self.torus_volume(self.torusRadiusMajor,
                                                     self.torusRadiusMinorInner))
        self.torusInnerVolume = self.torus_volume(self.torusRadiusMajor,
                                                  self.torusRadiusMinorInner) 
        self.torusInnerMass = self.torusInnerVolume * self.torusInnerRho
        self.torusOuterMass = self.torusOuterVolume * self.torusOuterRho
        self.torusInnerIxx = self.torus_ixx(self.torusInnerMass,
                                            self.torusRadiusMajor,
                                            self.torusRadiusMinorInner)
        self.torusInnerIyy = self.torusInnerIxx
        self.torusInnerIzz =  self.torus_izz(self.torusInnerMass,
                                             self.torusRadiusMajor,
                                             self.torusRadiusMinorInner)
        self.torusOuterIxx, self.torusOuterIyy, self.torusOuterIzz = (
            self.torus_hollow_ixxyyzz(self.torusOuterRho,
                                      self.torusRadiusMajor,
                                      self.torusRadiusMinor,
                                      self.torusRadiusMinorInner))
        self.torusMass = self.torusInnerMass + self.torusOuterMass
        self.torusIxx = self.torusInnerIxx + self.torusOuterIxx
        self.torusIyy = self.torusInnerIyy + self.torusOuterIyy
        self.torusIzz = self.torusInnerIzz + self.torusOuterIzz

        # combined disc & torus properties
        self.mass = self.discMass + self.torusMass
        self.Ixx = self.discIxx + self.torusIxx
        self.Iyy = self.discIyy + self.torusIyy
        self.Izz = self.discIzz + self.torusIzz

        # cost estimates
        self.discCost = self.discMass * materials[discMaterial][1]
        self.torusInnerCost = self.torusInnerMass * materials[torusInnerMaterial][1]
        self.torusOuterCost = self.torusOuterMass * materials[torusOuterMaterial][1]
        self.totalCost = self.discCost + self.torusInnerCost + self.torusOuterCost

    def torus_volume(self, rMajor, rMinor):
        return 2 * np.pi**2 * rMajor * rMinor**2

    def torus_ixx(self, mass, rMajor, rMinor):
        return (1.0/8.0) * mass * (4*rMajor**2 + 5*rMinor**2)

    def torus_izz(self, mass, rMajor, rMinor):
        return (1.0/4.0) * mass * (4*rMajor**2 + 3*rMinor**2)

    def torus_hollow_ixxyyzz(self, density, rMajor, rMinor, rMinorInner):
        volumeTotal = self.torus_volume(rMajor, rMinor)
        volumeInner = self.torus_volume(rMajor, rMinorInner)
        massTotal = volumeTotal * density
        massInner = volumeInner * density
        ixxTotal = self.torus_ixx(massTotal, rMajor, rMinor)
        ixxInner = self.torus_ixx(massInner, rMajor, rMinorInner)
        ixxOuter = ixxTotal - ixxInner
        iyyOuter = ixxOuter
        izzTotal = self.torus_izz(massTotal, rMajor, rMinor)
        izzInner = self.torus_izz(massInner, rMajor, rMinorInner)
        izzOuter = izzTotal - izzInner
        return ixxOuter, iyyOuter, izzOuter

    def write_report(self, filename=None):
        if filename==None:
            filename = f'./{self.meshName}.report'
        file = open(filename, 'w')
        currentTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file.write(f'Report for {self.meshName} generated @ {currentTime}\n\n')
        file.write(f'Disc radius (m): {self.discRadius:.2f}\n')
        file.write(f'Disc thickness (m): {self.discThickness:.2f}\n')
        file.write(f'Torus radius (m): {self.torusRadiusMinor:.2f}\n')
        file.write(f'Torus thickness (m): {self.torusThickness:.2f}\n')
        file.write(f'Damper Ixx (kg m^2): {self.Ixx:.3f}\n')
        file.write(f'Damper Iyy (kg m^2): {self.Iyy:.3f}\n')
        file.write(f'Damper Izz (kg m^2): {self.Izz:.3f}\n')
        file.write(f'Damper mass (kg): {self.mass:.3f}\n')
        file.write(f'Damper cost ($): {self.totalCost:.2f}\n')

def closed_hemisphere_xsection(hemisphereRadius, hemispherePoints):
    linSpacing = np.linspace(0, 90, hemispherePoints)
    cosineSpacing = np.cos(d2r*linSpacing) * (90*d2r)
    arcXPts = hemisphereRadius*np.sin(cosineSpacing)
    arcZPts = -hemisphereRadius*np.cos(cosineSpacing)
    arcZPts[0] = 0.0 # enforce = 0.0 @ waterline
    # for closed hemisphere (@ waterline):
    topXPts = np.cos(d2r*np.linspace(270,360,int(hemispherePoints/2)))*arcXPts[0]
    topZPts = np.full(len(topXPts), 0.0) # enforce = 0.0 (waterline)
    xPts = np.append(topXPts[:-1], arcXPts)
    zPts = np.append(topZPts[:-1], arcZPts)
    return xPts, zPts

def open_hemisphere_xsection(hemisphereRadius, hemispherePoints):
    linSpacing = np.linspace(0, 90, hemispherePoints)
    cosineSpacing = np.cos(d2r*linSpacing) * (90*d2r)
    arcXPts = hemisphereRadius*np.sin(cosineSpacing)
    arcZPts = -hemisphereRadius*np.cos(cosineSpacing)
    arcZPts[0] = 0.0 # enforce = 0.0 @ waterline
    return arcXPts, arcZPts

def stepped_cylinder_open_xsection(innCylRadius, innCylDraft, innCylPts,
                                   outCylRadius, outCylDraft, outCylPts,
                                   stepPts, bottomPts):
    '''create a 'stepped' cylinder cross section with open top @ waterline'''
    topCylXPts = np.full(innCylPts, innCylRadius)
    topCylZPts = create_cosSpace(length=innCylDraft, refining='both',
                                 pts=innCylPts)

    stepXPts = create_cosSpace(length=(outCylRadius-innCylRadius),
                                   refining='both', pts=stepPts)+innCylRadius
    stepZPts = np.full(stepPts, innCylDraft)

    outCylXPts = np.full(outCylPts, outCylRadius)
    outCylZPts = create_cosSpace(length=outCylDraft, refining='both',
                                 pts=outCylPts)+innCylDraft

    bottomXPts = create_cosSpace(length=outCylRadius, refining='end', pts=bottomPts)
    bottomZPts = np.full(bottomPts, (innCylDraft + outCylDraft))
    xPts = np.hstack((topCylXPts, stepXPts, outCylXPts, bottomXPts[::-1]))
    zPts = np.hstack((topCylZPts[::-1], stepZPts, outCylZPts[::-1], bottomZPts))
    pts = np.stack((xPts.T, zPts.T))
    return pts

def disc_with_torus_xsection(discRadius, discPoints, discThickness,
                             torusRadius, torusPoints):
    # define points along torus outer surface - in x-z cross-section
    xCentreTorus = discRadius + torusRadius
    zCentreTorus = 0.0
    # angleJoint: where torus meets disc, in degs
    angleJoint = np.sin((discThickness/2.0)/torusRadius) * r2d
    thetaTorus = np.linspace(-(90-angleJoint)*d2r, (270-angleJoint)*d2r, torusPoints)
    xTorus = xCentreTorus + torusRadius*np.sin(thetaTorus)
    zTorus = zCentreTorus + torusRadius*np.cos(thetaTorus)

    # define points along disc outer surface - in x-z cross-section
    linSpaceTop = np.linspace(270, 360, discPoints)
    cosSpaceTop = np.cos(d2r*linSpaceTop)
    xPlateTop = xTorus[0] * cosSpaceTop
    zPlateTop = np.full(len(xPlateTop), zTorus[0])
    linSpaceBottom = np.linspace(180, 270, discPoints)
    cosSpaceBottom = np.cos(d2r*linSpaceBottom)
    xPlateBottom = xTorus[-1] * cosSpaceBottom * -1
    zPlateBottom = np.full(len(xPlateBottom), zTorus[-1])

    # concatenate, remove duplicate points
    xPts = np.concatenate((xPlateTop[:-1], xTorus, xPlateBottom[1:]))
    zPts = np.concatenate((zPlateTop[:-1], zTorus, zPlateBottom[1:]))

    return xPts, zPts
