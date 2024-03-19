import numpy as np
import pygame as pg
from utilities import *
import os 
import time

path = os.path.dirname(os.path.realpath(__file__))
pg.init()
w,h = 1900,1000
screen = pg.display.set_mode((w,h), depth = True)
clock = pg.time.Clock()
running = True

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
                    verts.append(vec3D(float(coordinate[0]), float(coordinate[1]), float(coordinate[2])))
                except Exception:
                    verts.append(vec3D(float(coordinate[1]), float(coordinate[2]), float(coordinate[3])))
            
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

def fillTri(screen, x1, y1, x2, y2, x3, y3, col):
    pg.draw.polygon(screen, col, [[x1, y1], [x2, y2], [x3, y3]])

def drawTri(screen, x1, y1, x2, y2, x3, y3, col):
    pg.draw.line(screen, col, (x1, y1), (x2, y2))
    pg.draw.line(screen, col, (x2, y2), (x3, y3))
    pg.draw.line(screen, col, (x3, y3), (x1, y1))

#init construction of the mesh
mesh1 = Mesh(0, 0, 10, "/obj/VideoShip.obj")
figures = [mesh1]

#parameter
startHour = 6
tickRate = 10
font = pg.font.SysFont("Verdana", 20)


vLookFor, vLookSi, vCamera = vec3D(), vec3D(), vec3D()
last_time = time.time()
matProj = MatrixMakeProjection(90.0, h/w, 0.1, 1000)
fTheta, fYaw = 0.0, 0.0

while running:
    
    fElapsedTime = clock.tick(60) / 1000.0  # Elapsed time in seconds
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
        fYaw += 2.0 * fElapsedTime
    if keys[pg.K_d]:
        fYaw -= 2.0 * fElapsedTime

    fTheta += 1 * fElapsedTime
    matRotZ = MatrixRotationZ(fTheta)
    matRotX = MatrixRotationX(fTheta * 0.5)
    matRotY = MatrixRotationY(fTheta)

    matWorld = MatrixMakeIdentity()	# Form World Matrix
    matWorld = MatrixMultiplyMatrix(matRotZ, matRotX) # Transform by rotation
    #matWorld = MatrixMultiplyMatrix(matWorld, matRotY) # Transform by rotation

    # Create "Point At" Matrix for camera
    vUp = vec3D(0.0, 1.0, 0.0)
    vTarget = vec3D(0.0, 0.0, 1.0)
    vTarget2 = vec3D(1.0, 0.0, 0.0)
    matCameraRot = MatrixRotationY(fYaw)
    vLookFor = MatrixMultiplyVector(matCameraRot, vTarget)
    vLookSi = MatrixMultiplyVector(matCameraRot, vTarget2)
    vTarget = VectorAdd(vCamera, vLookFor)
    matView = MatrixPointAt(vCamera, vTarget, vUp)

    for mesh in figures:
        font = pg.font.SysFont("Verdana", 20)
        
        triangles = []
        for tri in mesh.mesh:
            
            triTransformed = Tri(MatrixMultiplyVector(matWorld, tri.p[0]),
                                 MatrixMultiplyVector(matWorld, tri.p[1]),
                                 MatrixMultiplyVector(matWorld, tri.p[2]))

            matTanslate = MatrixMakeTranslation(mesh.x, mesh.y, mesh.z)

            triTranslate = Tri(MatrixMultiplyVector(matTanslate, triTransformed.p[0]),
                                 MatrixMultiplyVector(matTanslate, triTransformed.p[1]),
                                 MatrixMultiplyVector(matTanslate, triTransformed.p[2]))

            line1 = VectorSub(triTranslate.p[1], triTranslate.p[0])
            line2 = VectorSub(triTranslate.p[2], triTranslate.p[0])

            # Take cross product of lines to get normal to triangle surface
            normal = VectorCrossProduct(line1, line2)

            # You normally need to normalise a normal!
            normal = VectorNormalise(normal)

            # Get Ray from triangle to camera
            vCameraRay = VectorSub(triTranslate.p[0], vCamera)

            if VectorDotProduct(normal, vCameraRay) < 0.0:
                hour = ((time.time() - last_time)/tickRate)%1
                light_information = soleil(vec3D(-1.0, 0.0, 0.0), hour)

                # Illumination
                light_direction = vec3D(light_information.x, light_information.y, light_information.z)
                light_direction = VectorNormalise(light_direction)

				# How similar is normal to light direction
            
                dp = max(0.1, VectorDotProduct(light_direction, VectorInverse(normal)))
                colour = GetColour(dp)
                
                # Convert World Space --> View Space 
                triViewed = Tri(MatrixMultiplyVector(matView, triTranslate.p[0]),
                                MatrixMultiplyVector(matView, triTranslate.p[1]),
                                MatrixMultiplyVector(matView, triTranslate.p[2]))
                triViewed.col = colour

                nClippedTriangles = 0
                clipped = [Tri(vec3D(),vec3D(),vec3D()),Tri(vec3D(),vec3D(),vec3D())]
                nClippedTriangles = triangleClipAgainstPlane(vec3D(0.0, 0.0, 0.1), vec3D(0.0, 0.0, 1.0), triViewed, clipped[0], clipped[1])

                for n in range(nClippedTriangles):

                    triProjected =  Tri(MatrixMultiplyVector(matProj, triViewed.p[0]),
                                        MatrixMultiplyVector(matProj, triViewed.p[1]),
                                        MatrixMultiplyVector(matProj, triViewed.p[2]))
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
        screen.fill("aqua")
        vertDrawn = 0

        for triangle in triangles:
            drawTri(screen, triangle.p[0].x, triangle.p[0].y, triangle.p[1].x, triangle.p[1].y, triangle.p[2].x, triangle.p[2].y, (0, 0, 0))
            fillTri(screen, triangle.p[0].x, triangle.p[0].y, triangle.p[1].x, triangle.p[1].y, triangle.p[2].x, triangle.p[2].y, triangle.col)
            vertDrawn += 1

    # blit info    
    fps = font.render(str(round(clock.get_fps())), True, pg.Color((255,0,0)))
    
    tim = font.render(str(round((hour*24 + startHour)%24, 4))+"h", True, pg.Color((255,0,0)))
    screen.blit(fps, (0,0))
    screen.blit(tim, (0,30))
    vertDrawn = font.render(str(vertDrawn) + " vert drawed", True, pg.Color((255,0,0)))
    screen.blit(vertDrawn, (50,0))
    ld = font.render("Light direction :", True, pg.Color((255,0,0)))
    
    pg.draw.line(screen, (255, 0, 0), (90, 140), (90+light_direction.x*50, 140+light_direction.y*50))
    screen.blit(ld, (0,60))
    pg.display.flip()
    clock.tick(60)

pg.quit()
