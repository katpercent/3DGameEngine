import numpy as np
import numba as nb
import pygame as pg
import math
import copy
import map
from collections import deque

import os 
import time
path = os.path.dirname(os.path.realpath(__file__))
pg.init()
w,h = 1900,1000
screen = pg.display.set_mode((w,h), depth = True)
clock = pg.time.Clock()
running = True

class vec3D:
    def __init__(self, x = 0, y = 0, z = 0, w = 1):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def arr(self):
        return [self.x, self.y, self.z]

class Tri:
    def __init__(self, p1 = vec3D(), p2 = vec3D(), p3 = vec3D()):
        self.p = [p1, p2, p3]
        self.col = (0, 0, 0)

class Mesh:
    def __init__(self, x, y, z, name="VideoShip.obj"):
        self.mesh = []
        verts = []
        vertsnormal = []
        vertstexture = []
        file = open(path+"/"+name, "r")
        file = file.readlines()
       
        for line in file:
            if line.startswith("v "):
                line = line.rstrip('\n')
                coordinate = line.split(" ")[1:]
                try :
                    verts.append(vec3D(float(coordinate[0]),float(coordinate[1]),float(coordinate[2])))
                except Exception:
                    verts.append(vec3D(float(coordinate[1]),float(coordinate[2]),float(coordinate[3])))
            
            elif line.startswith("vn "):
                line = line.rstrip('\n')
                coordinate = line.split(" ")[1:]
                try :
                    vertsnormal.append(vec3D(float(coordinate[0]),float(coordinate[1]),float(coordinate[2])))
                except Exception:
                    vertsnormal.append(vec3D(float(coordinate[1]),float(coordinate[2]),float(coordinate[3])))

            elif line.startswith("vt "):
                line = line.rstrip('\n')
                coordinate = line.split(" ")[1:]
                try :
                    vertstexture.append([float(coordinate[0]),float(coordinate[1])])
                except Exception:
                    vertstexture.append([float(coordinate[1]),float(coordinate[2])])
            
            elif line.startswith("f"):
                line = line.rstrip('\n')
                f = line.split(" ")[1:]
                if not("/" in line):
                    self.mesh.append(Tri(verts[int(f[0])-1],verts[int(f[1])-1],verts[int(f[2])-1]))
                else:
                    self.mesh.append(Tri(verts[int(f[0].split("/")[0])-1], verts[int(f[1].split("/")[0])-1], verts[int(f[2].split("/")[0])-1]))

        self.x = x 
        self.y = y
        self.z = z

class MeshCustom:
    def __init__(self, t1, t2):
        self.mesh = [
            t1,
            t2,                                               
        ]
        
    def change(self, t1, t2):
        self.mesh = [
            t1,
            t2,                                               
        ]

def MatrixMakeIdentity():
    matrix = [[0]*4 for _ in range(4)]
    matrix[0][0] = 1.0
    matrix[1][1] = 1.0
    matrix[2][2] = 1.0
    matrix[3][3] = 1.0
    return matrix

def MatrixMakeTranslation(x, y, z):
    matrix = [[0]*4 for _ in range(4)] 
    matrix[0][0] = 1.0
    matrix[1][1] = 1.0
    matrix[2][2] = 1.0
    matrix[3][3] = 1.0
    matrix[3][0] = x
    matrix[3][1] = y
    matrix[3][2] = z
    return matrix

def MatrixRotationX(fAngleRad):
    matrix = [[0]*4 for _ in range(4)] 
    matrix[0][0] = 1.0
    matrix[1][1] = math.cos(fAngleRad)
    matrix[1][2] = math.sin(fAngleRad)
    matrix[2][1] = -(math.sin(fAngleRad))
    matrix[2][2] = math.cos(fAngleRad)
    matrix[3][3] = 1.0
    return matrix
	
def MatrixRotationY(fAngleRad):
    matrix = [[0]*4 for _ in range(4)] 
    matrix[0][0] = math.cos(fAngleRad)
    matrix[0][2] = math.sin(fAngleRad)
    matrix[2][0] = -(math.sin(fAngleRad))
    matrix[1][1] = 1.0
    matrix[2][2] = math.cos(fAngleRad)
    matrix[3][3] = 1.0
    return matrix

def MatrixRotationZ(fAngleRad):
    matrix = [[0]*4 for _ in range(4)] 
    matrix[0][0] = math.cos(fAngleRad)
    matrix[0][1] = math.sin(fAngleRad)
    matrix[1][0] = -(math.sin(fAngleRad))
    matrix[1][1] = math.cos(fAngleRad)
    matrix[2][2] = 1.0
    matrix[3][3] = 1.0
    return matrix

def MatrixMakeProjection(fFovDegrees, fAspectRatio, fNear, fFar):
    fFovRad = 1.0 / math.tan(fFovDegrees * 0.5/ 180.0 * 3.14159)
    matrix = [[0]*4 for _ in range(4)] 
    matrix[0][0] = fAspectRatio * fFovRad
    matrix[1][1] = fFovRad
    matrix[2][2] = fFar / (fFar - fNear)
    matrix[3][2] = (-fFar * fNear) / (fFar - fNear)
    matrix[2][3] = 1.0
    return matrix

def MatrixQuickInverse(m): #Only for translation and Rotation Matrices
    matrix = MatrixMakeIdentity()

    matrix[0][0] = m[0][0]
    matrix[1][0] = m[0][1]
    matrix[2][0] = m[0][2]
    matrix[0][1] = m[1][0]
    matrix[1][1] = m[1][1]
    matrix[2][1] = m[1][2]
    matrix[0][2] = m[2][0]
    matrix[1][2] = m[2][1]
    matrix[2][2] = m[2][2]

    matrix[3][0] = -(m[3][0] * matrix[0][0] + m[3][1] * matrix[1][0] + m[3][2] * matrix[2][0])
    matrix[3][1] = -(m[3][0] * matrix[0][1] + m[3][1] * matrix[1][1] + m[3][2] * matrix[2][1])
    matrix[3][2] = -(m[3][0] * matrix[0][2] + m[3][1] * matrix[1][2] + m[3][2] * matrix[2][2])

    return matrix

def MatrixPointAt(pos, target, up):
    newForward = VectorSub(target, pos)
    newForward = VectorNormalise(newForward)

    a = VectorMul(newForward, VectorDotProduct(up, newForward))
    newUp = VectorSub(up, a)
    newUp = VectorNormalise(newUp)

    newRight = VectorCrossProduct(newUp, newForward)

    matrix = MatrixMakeIdentity()

    matrix[0][0] = newRight.x
    matrix[1][0] = newRight.y
    matrix[2][0] = newRight.z
    matrix[3][0] = -VectorDotProduct(pos, newRight)

    matrix[0][1] = newUp.x
    matrix[1][1] = newUp.y
    matrix[2][1] = newUp.z
    matrix[3][1] = -VectorDotProduct(pos, newUp)

    matrix[0][2] = newForward.x
    matrix[1][2] = newForward.y
    matrix[2][2] = newForward.z
    matrix[3][2] = -VectorDotProduct(pos, newForward)

    return matrix

def MatrixMultiplicationVector(m, i: vec3D):
    o = vec3D()
    o.x = i.x * m[0][0] + i.y * m[1][0] + i.z * m[2][0] + i.w * m[3][0]
    o.y = i.x * m[0][1] + i.y * m[1][1] + i.z * m[2][1] + i.w * m[3][1]
    o.z = i.x * m[0][2] + i.y * m[1][2] + i.z * m[2][2] + i.w * m[3][2]
    o.w = i.x * m[0][3] + i.y * m[1][3] + i.z * m[2][3] + i.w * m[3][3]

    return o

def vectorIntersectPlane(plane_p: vec3D, plane_n: vec3D, lineStart: vec3D, lineEnd: vec3D):
    plane_n = VectorNormalise(plane_n)
    plane_d = -(VectorDotProduct(plane_n, plane_p))
    ad = VectorDotProduct(lineStart, plane_n)
    bd = VectorDotProduct(lineEnd, plane_n)
    t = (-plane_d - ad) / (bd - ad)
    lineStartToEnd = VectorSub(lineEnd, lineStart)
    lineToIntersect = VectorMul(lineStartToEnd, t)
    return VectorAdd(lineStart, lineToIntersect)


def MatrixMultiplyMatrix(m1, m2):
    matrix = [[0 for _ in range(4)] for _ in range(4)]
    for c in range(4):
        for r in range(4):
            matrix[r][c] = m1[r][0] * m2[0][c] + m1[r][1] * m2[1][c] + m1[r][2] * m2[2][c] + m1[r][3] * m2[3][c]
    return matrix

def triangleClipAgainstPlane(plane_p: vec3D, plane_n: vec3D, in_tri: Tri, out_tri1: Tri, out_tri2: Tri) -> int:
    # Make sure plane normal is indeed normal
    plane_n = VectorNormalise(plane_n)

    # Return the dot product between a vector and a plane
    def dist(p: vec3D) -> float:
        n = VectorNormalise(p)
        return (plane_n.x * p.x + plane_n.y * p.y + plane_n.z * p.z - VectorDotProduct(plane_n, plane_p))

    # Copy the input triangle to the output triangle
    out_tri1 = copy.deepcopy(in_tri)
    out_tri2 = copy.deepcopy(in_tri)

    # Keep track of the number of triangles
    nClippedTriangles = 0

    # Calculate distances from each vertex of the triangle to the plane
    d0 = dist(in_tri.p[0])
    d1 = dist(in_tri.p[1])
    d2 = dist(in_tri.p[2])

    inside_points = []
    outside_points = []

    # Categorize the vertices based on their positions relative to the plane
    if d0 >= 0:
        inside_points.append(in_tri.p[0])
    else:
        outside_points.append(in_tri.p[0])
    if d1 >= 0:
        inside_points.append(in_tri.p[1])
    else:
        outside_points.append(in_tri.p[1])
    if d2 >= 0:
        inside_points.append(in_tri.p[2])
    else:
        outside_points.append(in_tri.p[2])

    # Case 1: All vertices are inside the plane
    if len(inside_points) == 3:
        out_tri1 = in_tri
        nClippedTriangles = 1

    # Case 2: One vertex is inside the plane
    elif len(inside_points) == 1 and len(outside_points) == 2:
        # The first triangle is formed by the inside vertex and the two intersection points
        out_tri1.p[0] = inside_points[0]
        out_tri1.p[1] = vectorIntersectPlane(plane_p, plane_n, inside_points[0], outside_points[0])
        out_tri1.p[2] = vectorIntersectPlane(plane_p, plane_n, inside_points[0], outside_points[1])
        nClippedTriangles = 1

    # Case 3: Two vertices are inside the plane
    elif len(inside_points) == 2 and len(outside_points) == 1:
        # The first triangle is formed by the two inside vertices and the intersection point
        out_tri1.p[0] = inside_points[0]
        out_tri1.p[1] = inside_points[1]
        out_tri1.p[2] = vectorIntersectPlane(plane_p, plane_n, inside_points[0], outside_points[0])

        # The second triangle is formed by one inside vertex, the intersection point, and the outside vertex
        out_tri2.p[0] = inside_points[0]
        out_tri2.p[1] = vectorIntersectPlane(plane_p, plane_n, inside_points[0], outside_points[0])
        out_tri2.p[2] = inside_points[1]

        nClippedTriangles = 2
    return 2


def VectorAdd(v1: vec3D, v2: vec3D):
	return vec3D(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)

def VectorSub(v1: vec3D, v2: vec3D):
	return vec3D(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z)

def VectorMul(v1: vec3D, k):
	return vec3D(v1.x * k, v1.y * k, v1.z * k)

def VectorMultiplyVector(v1: vec3D, v2: vec3D):
	return vec3D(v1.x * v2.x + v1.x * v2.y + v1.x * v2.z, v1.y * v2.x + v1.y * v2.y + v1.y * v2.z, v1.z * v2.x + v1.z * v2.y + v1.z * v2.z)

def VectorDiv(v1: vec3D, k):
    if k != 0:
	    return vec3D(v1.x / k, v1.y / k, v1.z / k)

def VectorDotProduct(v1: vec3D, v2: vec3D):
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def VectorLength(v: vec3D):
	return math.sqrt(VectorDotProduct(v, v))

def VectorNormalise(v: vec3D):
    l = VectorLength(v)
    if l != 0:
        return vec3D(v.x / l, v.y / l, v.z / l)
    return v

def VectorCrossProduct(v1: vec3D, v2: vec3D):
	return vec3D(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x)

def GetColour(lum):
    # Clamp luminosity between 0 and 1
    lum = max(0, min(1, lum))
    
    # Map luminosity to grayscale
    color_value = int(255 * lum)
    
    # Return grayscale color
    return color_value, color_value, color_value

def soleil(time):
    time_mapping = {
        4: [(-1, 0, 0), (255, 204, 153)],
        5: [(-1, 0, 0), (255, 225, 153)],
        6: [(-1, 1/3, 0), (255, 255, 153)],
        7: [(-1, 2/3, 0), (255, 246, 200)],
        8: [(-1, 1, 0), (240, 246, 230)],
        9: [(-1/3, 1, 0), (225, 246, 255)],
        10: [(-2/3, 1, 0), (205, 240, 255)],
        11: [(-1, 1, 0), (185, 235, 255)],
        12: [(-2/3, 1, 0), (165, 229, 255)],
        13: [(-1/3, 1, 0), (145, 224, 255)],
        14: [(0, 1, 0), (115, 215, 255)],
        15: [(1/3, 1, 0), (85, 198, 255)],
        16: [(2/3, 1, 0), (25, 189, 255)],
        17: [(1, 1, 0), (0, 171, 240)],
        18: [(1, 2/3, 0), (0, 45, 70)],
        19: [(1, 1/3, 0), (0, 20, 40)],
        20: [(1, 0, 0), (0, 10, 10)],
    }
    
    return time_mapping.get(time, [(0, 0, 0), (0, 0, 0)])

def fillTri(screen, x1, y1, x2, y2, x3, y3, col):
    pg.draw.polygon(screen, col, [[x1, y1], [x2, y2], [x3, y3]])

def drawTri(screen, x1, y1, x2, y2, x3, y3, col):
    pg.draw.line(screen, col, (x1, y1), (x2, y2))
    pg.draw.line(screen, col, (x2, y2), (x3, y3))
    pg.draw.line(screen, col, (x3, y3), (x1, y1))

start = time.time()
mesh1 = Mesh(0, -2, 0)
end = time.time()
ChargementTime = "Chargement time :" + str(end - start)
figures = [mesh1]
fTheta = 0.0
fYaw = 0.0
startHour = 6
vLookFor = vec3D()
vLookSi = vec3D()
vCamera = vec3D(0.0, 0.0, 0.0)
last_time = time.time()
matProj = MatrixMakeProjection(90.0, h/w, 0.1, 1000)
font = pg.font.SysFont("Verdana", 20)
# Buffer pour dessiner les arêtes du cube
buffer = pg.Surface((w, h))
lol = 0
while running:
    
    fElapsedTime = clock.tick(60) / 1000.0  # Elapsed time in seconds
    timeChargement = font.render(ChargementTime, True, pg.Color((0,0,0)))
    screen.blit(timeChargement, (0,70))
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    keys = pg.key.get_pressed()
    if keys[pg.K_UP]:
        vCamera.y += 8.0 * fElapsedTime
    if keys[pg.K_DOWN]:
        vCamera.y -= 8.0 * fElapsedTime
    if keys[pg.K_RIGHT]:
        vCamera = VectorSub(vCamera, vSide)
    if keys[pg.K_LEFT]:
        vCamera = VectorAdd(vCamera, vSide)
    
    vForward = VectorMul(vLookFor, 8.0 * fElapsedTime)
    vSide  = VectorMul(vLookSi, 8.0 * fElapsedTime)

    if keys[pg.K_z]:
        vCamera = VectorAdd(vCamera, vForward)
    if keys[pg.K_s]:
        vCamera = VectorSub(vCamera, vForward)
    if keys[pg.K_q]:
        fYaw -= 2.0 * fElapsedTime
    if keys[pg.K_d]:
        fYaw += 2.0 * fElapsedTime

    #fTheta += 1 * fElapsedTime
    matRotZ = MatrixRotationZ(fTheta)
    matRotX = MatrixRotationX(fTheta * 0.5)
    matTanslate = MatrixMakeTranslation(0.0, 0.0, 5)

    matWorld = MatrixMakeIdentity()	# Form World Matrix
    matWorld = MatrixMultiplyMatrix(matRotZ, matRotX) # Transform by rotation
    matWorld = MatrixMultiplyMatrix(matWorld, matTanslate) # Transform by translation

    # Create "Point At" Matrix for camera
    vUp = vec3D(0,1,0)
    vTarget = vec3D(0,0,1)
    vTarget2 = vec3D(1,0,0)
    matCameraRot = MatrixRotationY(fYaw)
    vLookFor = MatrixMultiplicationVector(matCameraRot, vTarget)
    vLookSi = MatrixMultiplicationVector(matCameraRot, vTarget2)
    vTarget = VectorAdd(vCamera, vLookFor)
    matView = MatrixPointAt(vCamera, vTarget, vUp)


    for mesh in figures:
        font = pg.font.SysFont("Verdana", 20)
        
        triangles = []
        for tri in mesh.mesh:
            
            matTanslate = MatrixMakeTranslation(mesh.x, mesh.y, mesh.z)
            matTri = MatrixMultiplyMatrix(matWorld, matTanslate)
            triTransformed = Tri(MatrixMultiplicationVector(matTri, tri.p[0]),
                                 MatrixMultiplicationVector(matTri, tri.p[1]),
                                 MatrixMultiplicationVector(matTri, tri.p[2]))

            line1 = VectorSub(triTransformed.p[1], triTransformed.p[0])
            line2 = VectorSub(triTransformed.p[2], triTransformed.p[0])

            # Take cross product of lines to get normal to triangle surface
            normal = VectorCrossProduct(line1, line2)

            # You normally need to normalise a normal!
            normal = VectorNormalise(normal)

            # Get Ray from triangle to camera
            vCameraRay = VectorSub(triTransformed.p[0], vCamera)

            if VectorDotProduct(normal, vCameraRay) < 0.0:

                hour = round((time.time() - last_time + startHour)%24)
                light_information = soleil(hour)

                # Illumination
                light_direction = vec3D(light_information[0][0], light_information[0][1], light_information[0][2])
                light_direction = VectorNormalise(light_direction)

				# How similar is normal to light direction
            
                dp = max(0.1, VectorDotProduct(light_direction, normal))
                colour = GetColour(dp)
                
                # Convert World Space --> View Space
                triViewed = Tri(MatrixMultiplicationVector(matView, triTransformed.p[0]),
                                MatrixMultiplicationVector(matView, triTransformed.p[1]),
                                MatrixMultiplicationVector(matView, triTransformed.p[2]))
                triViewed.col = colour

                nClippedTriangles = 0
                clipped = [Tri(vec3D(),vec3D(),vec3D()),Tri(vec3D(),vec3D(),vec3D())]
                nClippedTriangles = triangleClipAgainstPlane(vec3D(0.0, 0.0, 0.1), vec3D(0.0, 0.0, 1.0), triViewed, clipped[0], clipped[1])

                for n in range(nClippedTriangles):

                    triProjected =  Tri(MatrixMultiplicationVector(matProj, triViewed.p[0]),
                                        MatrixMultiplicationVector(matProj, triViewed.p[1]),
                                        MatrixMultiplicationVector(matProj, triViewed.p[2]))
                    triProjected.col = clipped[n].col
                
                    triProjected.p[0] = VectorDiv(triProjected.p[0], triProjected.p[0].w)
                    triProjected.p[1] = VectorDiv(triProjected.p[1], triProjected.p[1].w)
                    triProjected.p[2] = VectorDiv(triProjected.p[2], triProjected.p[2].w)
                    triProjected.col = colour

                    # X/Y are inverted so put them back
                    for j in range(3):
                        triProjected.p[j].x *= -1.0
                        triProjected.p[j].y *= -1.0
                    
                    vOffsetView = vec3D(1.0, 1.0, 0.0)
                    for i in range(3):
                        triProjected.p[i] = VectorAdd(triProjected.p[i], vOffsetView)
                        triProjected.p[i].x *= 0.5 * w
                        triProjected.p[i].y *= 0.5 * h
                
                    triangles.append(triProjected)

        triangles.sort(key=lambda t: (t.p[0].z + t.p[1].z + t.p[2].z) / 3.0, reverse=True)
        print(light_information)
        screen.fill(light_information[1])
        vertDrawn = 0

        for triangle in triangles:
            states = [not(triangle.p[i].x > w or triangle.p[i].x < 0 or triangle.p[i].y > h or triangle.p[i].y < 0) for i in range(3)]
            for state in states:
                if state:
                    drawTri(screen, triangle.p[0].x, triangle.p[0].y, triangle.p[1].x, triangle.p[1].y, triangle.p[2].x, triangle.p[2].y, (0, 0, 0))
                    fillTri(screen, triangle.p[0].x, triangle.p[0].y, triangle.p[1].x, triangle.p[1].y, triangle.p[2].x, triangle.p[2].y, triangle.col)
                    vertDrawn += 1
            
    fps = font.render(str(round(clock.get_fps())), True, pg.Color((255,0,0)))
    
    tim = font.render(str(hour)+"h", True, pg.Color((255,0,0)))
    screen.blit(fps, (0,0))
    screen.blit(tim, (0,30))
    vertDrawn = font.render(str(vertDrawn)+" vert drawed", True, pg.Color((255,0,0)))
    screen.blit(vertDrawn, (50,0))
    pg.display.flip()
    clock.tick(60)

pg.quit()
