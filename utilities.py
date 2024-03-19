import numpy as np
import math
import copy

class vec3D:
    def __init__(self, x = 0.0, y = 0.0, z = 0.0, w = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def arr(self):
        return np.array([self.x, self.y, self.z, self.w], dtype= float)
    
    def arrToList(self, arr):
        self.x, self.y, self.z, self.w = arr[0], arr[1], arr[2], arr[3]
        return self

class Tri:
    def __init__(self, p1 = vec3D(), p2 = vec3D(), p3 = vec3D()):
        self.p = [p1, p2, p3]
        self.col = (0, 0, 0)

def MatrixMakeIdentity():
    matrix = np.zeros((4,4), dtype=float)
    matrix[0][0] = 1.0
    matrix[1][1] = 1.0
    matrix[2][2] = 1.0
    matrix[3][3] = 1.0
    return matrix

def MatrixMakeTranslation(x, y, z):
    matrix = np.zeros((4,4), dtype=float)
    matrix[0][3] = x
    matrix[1][3] = y
    matrix[2][3] = z
    matrix[0][0] = 1.0
    matrix[1][1] = 1.0
    matrix[2][2] = 1.0
    matrix[3][3] = 1.0
    return matrix

def MatrixRotationX(fAngleRad):
    """
    [[0, 0, 0, 0]
    [0, 0, 0, 0]
    [0, 0, 0, 0]
    [0, 0, 0, 0]]
    """
    matrix = np.zeros((4,4), dtype=float)
    matrix[0][0] = 1.0
    matrix[1][1] = math.cos(fAngleRad)
    matrix[1][2] = -(math.sin(fAngleRad))
    matrix[2][1] = math.sin(fAngleRad)
    matrix[2][2] = math.cos(fAngleRad)
    matrix[3][3] = 1.0
    return matrix
	
def MatrixRotationY(fAngleRad):
    matrix = np.zeros((4,4), dtype=float)
    matrix[0][0] = math.cos(fAngleRad)
    matrix[0][2] = math.sin(fAngleRad)
    matrix[2][0] = -(math.sin(fAngleRad))
    matrix[1][1] = 1.0
    matrix[2][2] = math.cos(fAngleRad)
    matrix[3][3] = 1.0
    return matrix

def MatrixRotationZ(fAngleRad):
    """
    [[cos(deg), sin(deg), 0, 0]
    [-sin(deg), cos(deg), 0, 0]
    [     0   ,    0    , 1, 0]
    [     0   ,    0    , 0, 1]]
    """
    matrix = np.zeros((4,4), dtype=float)
    matrix[0][0] = math.cos(fAngleRad)
    matrix[0][1] = -(math.sin(fAngleRad))
    matrix[1][0] = math.sin(fAngleRad)
    matrix[1][1] = math.cos(fAngleRad)
    matrix[2][2] = 1.0
    matrix[3][3] = 1.0
    return matrix

def MatrixMakeProjection(fFovDegrees, fAspectRatio, fNear, fFar):
    fFovRad = 1.0 / math.tan(fFovDegrees * 0.5/ 180.0 * 3.14159)
    matrix = np.zeros((4,4), dtype=float)
    matrix[0][0] = fAspectRatio * fFovRad
    matrix[1][1] = fFovRad
    matrix[2][2] = fFar / (fFar - fNear)
    matrix[2][3] = (-fFar * fNear) / (fFar - fNear)
    matrix[3][2] = 1.0
    return matrix

def MatrixPointAt(pos, target, up):
    """
    [[newRightx, newUpx, newForwardx, 0.0]
    [newRighty, newUpy, newForwardy, 0.0]
    [newRightz, newUpz, newForwardz, 0.0]
    [dp1, dp2, dp3, 0.0]]
    """

    newForward = VectorSub(target, pos)
    newForward = VectorNormalise(newForward)

    a = VectorMul(newForward, VectorDotProduct(up, newForward))
    newUp = VectorSub(up, a)
    newUp = VectorNormalise(newUp)

    newRight = VectorCrossProduct(newUp, newForward)

    matrix = MatrixMakeIdentity()

    matrix[0][0] = newRight.x
    matrix[0][1] = newRight.y
    matrix[0][2] = newRight.z
    matrix[0][3] = -VectorDotProduct(pos, newRight)

    matrix[1][0] = newUp.x
    matrix[1][1] = newUp.y
    matrix[1][2] = newUp.z
    matrix[1][3] = -VectorDotProduct(pos, newUp)

    matrix[2][0] = newForward.x
    matrix[2][1] = newForward.y
    matrix[2][2] = newForward.z
    matrix[2][3] = -VectorDotProduct(pos, newForward)

    return matrix

def MatrixMultiplyVector(m, i: vec3D):
    return vec3D().arrToList(np.dot(m, i.arr()))

def MatrixMultiplyMatrix(m1, m2):
    return np.dot(m1, m2)

def vectorIntersectPlane(plane_p: vec3D, plane_n: vec3D, lineStart: vec3D, lineEnd: vec3D):
    plane_n = VectorNormalise(plane_n)
    plane_d = -(VectorDotProduct(plane_n, plane_p))
    ad = VectorDotProduct(lineStart, plane_n)
    bd = VectorDotProduct(lineEnd, plane_n)
    t = (-plane_d - ad) / (bd - ad)
    lineStartToEnd = VectorSub(lineEnd, lineStart)
    lineToIntersect = VectorMul(lineStartToEnd, t)
    return VectorAdd(lineStart, lineToIntersect)


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
    return nClippedTriangles


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

def VectorInverse(v: vec3D):
    return VectorMul(v, -1)

def GetColour(lum):
    # Clamp luminosity between 0 and 1
    lum = max(0, min(1, lum))
    
    # Map luminosity to grayscale
    color_value = int(255 * lum)
    
    # Return grayscale color
    return color_value, color_value, color_value

def soleil(vec: vec3D, time):
    matRot = MatrixRotationZ(math.radians(time * 360))
    return vec3D().arrToList(np.dot(matRot, vec.arr()))
