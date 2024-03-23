import time
import pygame as pg
import os 
from utilities import *

path = os.path.dirname(os.path.realpath(__file__))
pg.init()
w,h = 1900,1000
screen = pg.display.set_mode((w,h))
clock = pg.time.Clock()
running = True

class Mesh:
    def __init__(self, x, y, z, name="/obj/VideoShip.obj"):
        self.mesh = []
        verts = []
        file = open(path+"/"+name, "r").readlines()
        for line in file:
            if line.startswith("v "):
                coordinate = line.split(" ")[1:]
                if not("/" in line):
                    verts.append(vec3D(float(coordinate[0]), float(coordinate[1]), float(coordinate[2])))
                else:
                    verts.append(vec3D(float(coordinate[1]), float(coordinate[2]), float(coordinate[3])))
            
            elif line.startswith("f"):
                f = line.split(" ")[1:]
                if not("/" in line):
                    self.mesh.append(Tri(verts[int(f[0])-1],verts[int(f[1])-1],verts[int(f[2])-1]))
                else:
                    self.mesh.append(Tri(verts[int(f[0].split("/")[0])-1], verts[int(f[1].split("/")[0])-1], verts[int(f[2].split("/")[0])-1]))

        self.x = x 
        self.y = y
        self.z = z

#init construction of the mesh
mesh1 = Mesh(0.0, -15.0, 0.0, "/obj/Mountains.obj")
figures = [mesh1]

#parameter
startHour = 6
tickRate = 10
font = pg.font.SysFont("Verdana", 20)

#base variable
vLookFor, vLookSi, vCamera = vec3D(), vec3D(), vec3D()
matProj = MatrixMakeProjection(90.0, h/w, 0.1, 1000)
fTheta, fYaw = 0.0, 0.0
start = time.time()

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
    
    if keys[pg.K_ESCAPE]:
        quit()

    #fTheta += 1 * fElapsedTime
    matRotZ = MatrixRotationZ(fTheta)
    matRotX = MatrixRotationX(fTheta * 0.5)
    matRotY = MatrixRotationY(fTheta)

    matWorld = MatrixMakeIdentity()	# Form World Matrix
    matWorld = MatrixMultiplyMatrix(matRotZ, matRotX) # Transform by rotation ZX
    #matWorld = MatrixMultiplyMatrix(matWorld, matRotY) # Transform by rotation ZXY

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
        triangles2raster = []
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
            normal = VectorNormalise(normal)

            # Get Ray from triangle to camera
            vCameraRay = VectorSub(triTranslate.p[0], vCamera)

            if VectorDotProduct(normal, vCameraRay) < 0.0:
                hour = ((time.time() - start)/tickRate)%1
                light_information = soleil(vec3D(-1.0, 0.0, 0.0), hour)

                # Illumination
                light_direction = vec3D(light_information.x, light_information.y, light_information.z)
                light_direction = VectorNormalise(light_direction)

				# How similar is normal to light direction
                colour = GetColour(max(0.1, VectorDotProduct(light_direction, VectorInverse(normal))))
                
                # Convert World Space --> View Space 
                triViewed = Tri(MatrixMultiplyVector(matView, triTranslate.p[0]),
                                MatrixMultiplyVector(matView, triTranslate.p[1]),
                                MatrixMultiplyVector(matView, triTranslate.p[2]))

                nClippedTriangles, clipped = triangleClipAgainstPlane(vec3D(0.0, 0.0, 0.1), vec3D(0.0, 0.0, 1.0), triViewed)

                for n in range(nClippedTriangles):
                    triProjected =  Tri(MatrixMultiplyVector(matProj, clipped[n].p[0]),
                                        MatrixMultiplyVector(matProj, clipped[n].p[1]),
                                        MatrixMultiplyVector(matProj, clipped[n].p[2]))
                    
                    triProjected.p[0] = VectorDiv(triProjected.p[0], triProjected.p[0].w)
                    triProjected.p[1] = VectorDiv(triProjected.p[1], triProjected.p[1].w)
                    triProjected.p[2] = VectorDiv(triProjected.p[2], triProjected.p[2].w)
                    triProjected.col = colour

                    vOffsetView = vec3D(1.0, 1.0, 0.0)
                    for i in range(3):
                        triProjected.p[i].x *= -1.0
                        triProjected.p[i].y *= -1.0
                        triProjected.p[i] = VectorAdd(triProjected.p[i], vOffsetView)
                        triProjected.p[i].x *= 0.5 * w
                        triProjected.p[i].y *= 0.5 * h
                
                    triangles2raster.append(triProjected)

        triangles2raster.sort(key=lambda t: (t.p[0].z + t.p[1].z + t.p[2].z) / 3.0, reverse=True)
        screen.fill("aqua")

        triDraw = 0
        # Clip triangles against all four screen edges
        for triangle in triangles2raster:
            clippedTriangles = [triangle]
            trianglesToAdd = []
            for plane in range(4):
                if plane == 0:
                    # Clip against left edge
                    trianglesToAdd = []
                    for t in clippedTriangles:
                        nClippedTriangles, clipped = triangleClipAgainstPlane(vec3D(0.0, 0.0, 0.0), vec3D(1.0, 0.0, 0.0), t)
                        trianglesToAdd += clipped[:nClippedTriangles]
                    clippedTriangles = list(trianglesToAdd)
                elif plane == 1:
                    # Clip against right edge
                    trianglesToAdd = []
                    for t in clippedTriangles:
                        nClippedTriangles, clipped = triangleClipAgainstPlane(vec3D(w - 1, 0.0, 0.0), vec3D(-1.0, 0.0, 0.0), t)
                        trianglesToAdd += clipped[:nClippedTriangles]
                    clippedTriangles = list(trianglesToAdd)
                elif plane == 2:
                    # Clip against top edge
                    trianglesToAdd = []
                    for t in clippedTriangles:
                        nClippedTriangles, clipped = triangleClipAgainstPlane(vec3D(0.0, 0.0, 0.0), vec3D(0.0, 1.0, 0.0), t)
                        trianglesToAdd += clipped[:nClippedTriangles]
                    clippedTriangles = list(trianglesToAdd)
                elif plane == 3:
                    # Clip against bottom edge
                    trianglesToAdd = []
                    for t in clippedTriangles:
                        nClippedTriangles, clipped = triangleClipAgainstPlane(vec3D(0.0, h - 1, 0.0), vec3D(0.0, -1.0, 0.0), t)
                        trianglesToAdd += clipped[:nClippedTriangles]
                    clippedTriangles = list(trianglesToAdd)
            # Draw the triangles
            for triangle in clippedTriangles:
                drawTri(screen, triangle.p[0].x, triangle.p[0].y, triangle.p[1].x, triangle.p[1].y, triangle.p[2].x, triangle.p[2].y, (255, 0, 0))
                fillTri(screen, triangle.p[0].x, triangle.p[0].y, triangle.p[1].x, triangle.p[1].y, triangle.p[2].x, triangle.p[2].y, triangle.col)
                triDraw += 1

    # blit info    
    fps = font.render(str(round(clock.get_fps())), True, pg.Color((255,0,0)))
    tim = font.render(str(round((hour*24 + startHour)%24, 4))+"h", True, pg.Color((255,0,0)))
    lightdir = font.render("Light direction :", True, pg.Color((255,0,0)))
    triDrawn = font.render("Triangles Drawn : " +  str(triDraw), True, pg.Color((255,0,0)))
    pg.draw.line(screen, (255, 0, 0), (90, 140), (90 + light_direction.x * 50, 140 + light_direction.y * 50))
    screen.blit(lightdir, (0,60)), screen.blit(fps, (0,0)), screen.blit(tim, (0,30)), screen.blit(triDrawn, (60,0))
    pg.display.flip()
    clock.tick(60)

pg.quit()
