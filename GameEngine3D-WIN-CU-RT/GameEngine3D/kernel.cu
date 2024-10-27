
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <array>
#include <cmath>
#include<map>
#include "wtypes.h"
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<sstream>
#include<algorithm>
#include <chrono>
#include <SDL.h>

struct vec4
{
    float coo[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
};
class rgb {
public:
    float color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
};
struct vec3
{
    float coo[3] = { 0.0, 0.0, 1.0 };
};
struct triangle
{
    vec4 p[3];
    vec3 t[3];
};
struct mesh {
    //const char* textureFileName = "texture.png";
    //const char* textureLocName = path.c_str();
    //const char* FinalTexturePath = textureLocName + textureFileName;
    std::vector<triangle> tris;
    int trianglesSize = tris.size();
    //SDL_Surface* image = SDL_LoadBMP_RW(SDL_RWFromFile(, "rb"), 1);
    std::vector<vec3> textures;
    vec4 coo;
    vec4 rot;

    bool LoadFromObjectFile(const std::string& sFilename, bool HasTexture = false) {
        std::ifstream f(sFilename);
        if (!f.is_open()) {
            std::cerr << "Failed to open file: " << sFilename << std::endl;
            return false;
        }

        // Local cache of vertices
        std::vector<vec4> verts;

        while (!f.eof()) {
            char line[128];
            f.getline(line, 128);

            std::stringstream s;
            s << line;

            char junk;

            if (line[0] == 'v' && line[1] == ' ') {
                vec4 v;
                s >> junk >> v.coo[0] >> v.coo[1] >> v.coo[2];
                verts.push_back(v);
            }
            else if (line[0] == 'v' && line[1] == 't')
            {
                vec3 v;
                s >> junk >> junk >> v.coo[0] >> v.coo[1];
                // A little hack for the spyro texture
                //v.u = 1.0f - v.u;
                //v.v = 1.0f - v.v;
                textures.push_back(v);
            }
            else if (line[0] == 'f' && line[1] == ' ') {
                if (!HasTexture)
                {
                    std::vector<int> f;
                    std::string segment;
                    while (s >> segment) {
                        if (segment != "f") {
                            std::stringstream segmentStream(segment);
                            std::string indexStr;
                            getline(segmentStream, indexStr, '/');
                            f.push_back(stoi(indexStr) - 1);
                        }
                    }

                    if (f.size() == 3) {
                        tris.push_back({ verts[f[0]], verts[f[1]], verts[f[2]] });
                    }
                    else if (f.size() > 3) {
                        for (size_t i = 1; i < f.size() - 1; ++i) {
                            tris.push_back({ verts[f[0]], verts[f[i]], verts[f[i + 1]] });
                        }
                    }
                }
                else
                {
                    s >> junk;

                    std::string tokens[6];
                    int nTokenCount = -1;


                    while (!s.eof())
                    {
                        char c = s.get();
                        if (c == ' ' || c == '/')
                            nTokenCount++;
                        else
                            tokens[nTokenCount].append(1, c);
                    }

                    tokens[nTokenCount].pop_back();

                    tris.push_back({ verts[stoi(tokens[0]) - 1], verts[stoi(tokens[2]) - 1], verts[stoi(tokens[4]) - 1], textures[stoi(tokens[1]) - 1], textures[stoi(tokens[3]) - 1], textures[stoi(tokens[5]) - 1] });

                }
            }

        }
        trianglesSize = tris.size();
        return true;
    }
    // Method to scale the mesh
    void Scale(float scaleFactor) {
        for (auto& tri : tris) {
            for (auto& vertex : tri.p) {
                vertex.coo[0] *= scaleFactor;
                vertex.coo[1] *= scaleFactor;
                vertex.coo[2] *= scaleFactor;
            }
        }
    }

    // Method to scale the mesh
    void Pos(float x, float y, float z) {
        for (auto& tri : tris) {
            for (auto& vertex : tri.p) {
                vertex.coo[0] += x;
                vertex.coo[1] += y;
                vertex.coo[2] += z;
            }
        }
    }

    // Method to change the position of the mesh
    void ChangePos(float x, float y, float z) {
        coo.coo[0] = x;
        coo.coo[1] = y;
        coo.coo[2] = z;
    }
};
struct meshes
{
    std::vector<mesh> meshes;
};
struct mat4x4
{
    float m[4][4] = { 0 };
};

struct ray
{
    vec4 p1;
    vec4 p2;
    vec4 lab = vec4({ p1.coo[0] - p1.coo[0], p1.coo[1] - p1.coo[1], p1.coo[2] - p1.coo[2] });
};

struct camera
{
    vec4 cameraPosition;
    vec4 cameraLookAt;
    vec4 cameraUp;
    float velocity;
};
struct Light
{
    vec4 lightPosition;
    vec4 lightColor;
    float radius;
};

// Get the horizontal and vertical screen sizes in pixel
void GetDesktopResolution(int& horizontal, int& vertical)
{
    RECT desktop;
    // Get a handle to the desktop window
    const HWND hDesktop = GetDesktopWindow();
    // Get the size of screen to the variable desktop
    GetWindowRect(hDesktop, &desktop);
    // The top left corner will have coordinates (0,0)
    // and the bottom right corner will have coordinates
    // (horizontal, vertical)
    horizontal = desktop.right - 100;
    vertical = desktop.bottom - 100;
};

__host__ __device__ vec4 RgbMultiplication(const vec4& a, const vec4& b) {
    vec4 result;
    result.coo[0] = a.coo[0] * b.coo[0];
    result.coo[1] = a.coo[1] * b.coo[1];
    result.coo[2] = a.coo[2] * b.coo[2];
    return result;
}

__host__ __device__ vec4 VectorSubtract(vec4& a, vec4& b) {
    vec4 result;
    result.coo[0] = a.coo[0] - b.coo[0];
    result.coo[1] = a.coo[1] - b.coo[1];
    result.coo[2] = a.coo[2] - b.coo[2];
    return result;
}

__host__ __device__ float VectorDotProduct(vec4 v1, vec4 v2)
{
    float result = v1.coo[0] * v2.coo[0] + v1.coo[1] * v2.coo[1] + v1.coo[2] * v2.coo[2];
    return result;
};

__host__ __device__ vec4 VectorAddition(vec4& a, vec4& b)
{
    vec4 result;
    result.coo[0] = a.coo[0] + b.coo[0];
    result.coo[1] = a.coo[1] + b.coo[1];
    result.coo[2] = a.coo[2] + b.coo[2];
    return result;
}; // V

__host__ __device__ vec4 VectorMultiplication(vec4& a, float& k)
{
    vec4 result;
    result.coo[0] = a.coo[0] * k;
    result.coo[1] = a.coo[1] * k;
    result.coo[2] = a.coo[2] * k;
    return result;
}; // V

__host__ __device__ vec4 VectorDivision(vec4& a, float& k)
{
    vec4 result;
    if (k != 0) {
        result.coo[0] = a.coo[0] / k;
        result.coo[1] = a.coo[1] / k;
        result.coo[2] = a.coo[2] / k;
    }
    return result;
}; // V

__host__ __device__ vec4 VectorMultiplyVector(vec4& a, vec4& b)
{
    vec4 result;
    result.coo[0] = a.coo[0] * b.coo[0] + a.coo[0] * b.coo[1] + a.coo[0] * b.coo[2];
    result.coo[1] = a.coo[1] * b.coo[0] + a.coo[1] * b.coo[1] + a.coo[1] * b.coo[2];
    result.coo[2] = a.coo[2] * b.coo[0] + a.coo[2] * b.coo[1] + a.coo[2] * b.coo[2];
    return result;
};

__host__ __device__ float VectorLength(vec4& v)
{
    return sqrtf(VectorDotProduct(v, v));
};

__host__ __device__ vec4 VectorNormalize(vec4& v)
{
    float l = VectorLength(v);
    vec4 result;
    if (l != 0) {
        result.coo[0] = v.coo[0] / l; result.coo[1] = v.coo[1] / l; result.coo[2] = v.coo[2] / l;
    }
    return result;
};

__host__ __device__ vec4 VectorCrossProduct(vec4& v1, vec4& v2)
{
    vec4 result;
    result.coo[0] = v1.coo[1] * v2.coo[2] - v1.coo[2] * v2.coo[1];
    result.coo[1] = v1.coo[2] * v2.coo[0] - v1.coo[0] * v2.coo[2];
    result.coo[2] = v1.coo[0] * v2.coo[1] - v1.coo[1] * v2.coo[0];
    return result;
};

__host__ __device__ vec4 VectorInverse(vec4& v)
{
    v.coo[0] *= -1; v.coo[1] *= -1; v.coo[2] *= -1;
    return v;
};

__host__ __device__ mat4x4 MatrixMakeIdentity()
{
    mat4x4 matrix;
    matrix.m[0][0] = 1.0;
    matrix.m[1][1] = 1.0;
    matrix.m[2][2] = 1.0;
    matrix.m[3][3] = 1.0;
    return matrix;
};

__host__ __device__ void MatrixMultiplyVector(vec4& o, mat4x4& m, vec4& i)
{
    o.coo[0] = i.coo[0] * m.m[0][0] + i.coo[1] * m.m[0][1] + i.coo[2] * m.m[0][2] + m.m[0][3];
    o.coo[1] = i.coo[0] * m.m[1][0] + i.coo[1] * m.m[1][1] + i.coo[2] * m.m[1][2] + m.m[1][3];
    o.coo[2] = i.coo[0] * m.m[2][0] + i.coo[1] * m.m[2][1] + i.coo[2] * m.m[2][2] + m.m[2][3];
    o.coo[3] = i.coo[0] * m.m[3][0] + i.coo[1] * m.m[3][1] + i.coo[2] * m.m[3][2] + m.m[3][3];

};

__host__ __device__ mat4x4 MatrixMultiplyMatrix(mat4x4& m1, mat4x4& m2)
{
    mat4x4 matrix;
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            matrix.m[r][c] = m1.m[r][0] * m2.m[0][c] + m1.m[r][1] * m2.m[1][c] + m1.m[r][2] * m2.m[2][c] + m1.m[r][3] * m2.m[3][c];
    return matrix;
}

__host__ __device__ mat4x4 MatrixTranslation(vec4 coordinates)
{
    mat4x4 matrix;
    matrix.m[0][3] = coordinates.coo[0];
    matrix.m[1][3] = coordinates.coo[1];
    matrix.m[2][3] = coordinates.coo[2];
    matrix.m[0][0] = 1.0;
    matrix.m[1][1] = 1.0;
    matrix.m[2][2] = 1.0;
    matrix.m[3][3] = 1.0;
    return matrix;
}

__host__ __device__ mat4x4 MatrixPointAt(vec4& pos, vec4& target, vec4& up)
{

    /* [[newRightx, newUpx, newForwardx, 0.0]
    [newRighty, newUpy, newForwardy, 0.0]
    [newRightz, newUpz, newForwardz, 0.0]
    [dp1, dp2, dp3, 0.0]]*/

    vec4 newForward = VectorSubtract(target, pos);
    newForward = VectorNormalize(newForward);

    float r = VectorDotProduct(up, newForward);
    vec4 a = VectorMultiplication(newForward, r);
    vec4 newUp = VectorSubtract(up, a);
    newUp = VectorNormalize(newUp);

    vec4 newRight = VectorCrossProduct(newUp, newForward);

    mat4x4 matrix = MatrixMakeIdentity();

    matrix.m[0][0] = newRight.coo[0];
    matrix.m[0][1] = newRight.coo[1];
    matrix.m[0][2] = newRight.coo[2];
    matrix.m[0][3] = -VectorDotProduct(pos, newRight);

    matrix.m[1][0] = newUp.coo[0];
    matrix.m[1][1] = newUp.coo[1];
    matrix.m[1][2] = newUp.coo[2];
    matrix.m[1][3] = -VectorDotProduct(pos, newUp);

    matrix.m[2][0] = newForward.coo[0];
    matrix.m[2][1] = newForward.coo[1];
    matrix.m[2][2] = newForward.coo[2];
    matrix.m[2][3] = -VectorDotProduct(pos, newForward);
    return matrix;
};

__host__ __device__ mat4x4 MatrixRotationX(float& AngleRad)
{
    mat4x4 matrix;
    matrix.m[0][0] = 1.0;
    matrix.m[1][1] = cosf(AngleRad);
    matrix.m[1][2] = -(sinf(AngleRad));
    matrix.m[2][1] = sinf(AngleRad);
    matrix.m[2][2] = cosf(AngleRad);
    matrix.m[3][3] = 1.0;
    return matrix;
};

__host__ __device__ mat4x4 MatrixRotationY(float& AngleRad)
{
    mat4x4 matrix;
    matrix.m[0][0] = cosf(AngleRad);
    matrix.m[0][2] = sinf(AngleRad);
    matrix.m[2][0] = -(sinf(AngleRad));
    matrix.m[1][1] = 1.0;
    matrix.m[2][2] = cosf(AngleRad);
    matrix.m[3][3] = 1.0;
    return matrix;
};

__host__ __device__ mat4x4 MatrixRotationZ(float& AngleRad)
{
    mat4x4 matrix;
    matrix.m[0][0] = cosf(AngleRad);
    matrix.m[0][1] = -(sinf(AngleRad));
    matrix.m[1][0] = sinf(AngleRad);
    matrix.m[1][1] = cosf(AngleRad);
    matrix.m[2][2] = 1.0;
    matrix.m[3][3] = 1.0;
    return matrix;
};

__host__ __device__ vec4 Vector_IntersectPlane(vec4& plane_p, vec4& plane_n, vec4& lineStart, vec4& lineEnd, float& t)
{
    plane_n = VectorNormalize(plane_n);
    float plane_d = -VectorDotProduct(plane_n, plane_p);
    float ad = VectorDotProduct(lineStart, plane_n);
    float bd = VectorDotProduct(lineEnd, plane_n);
    t = (-plane_d - ad) / (bd - ad);
    vec4 lineStartToEnd = VectorSubtract(lineEnd, lineStart);
    vec4 lineToIntersect = VectorMultiplication(lineStartToEnd, t);
    return VectorAddition(lineStart, lineToIntersect);;
}

__host__ __device__ int maxValue(int a, int b) {
    if (a > b)
        return a;
    else return b;
};

__host__ __device__ int minValue(int a, int b) {
    if (a > b)
        return b;
    else return a;
};

__device__ bool ray_intersects_triangles(vec4& ray_origin, vec4& ray_vector, triangle& triangle, vec4& intersection)
{
    constexpr float epsilon = FLT_EPSILON;

    vec4 edge1 = VectorSubtract(triangle.p[1], triangle.p[0]);
    vec4 edge2 = VectorSubtract(triangle.p[2], triangle.p[0]);
    vec4 ray_cross_e2 = VectorCrossProduct(ray_vector, edge2);
    float det = VectorDotProduct(edge1, ray_cross_e2);

    if (det > -epsilon && det < epsilon)
        return false;    // This ray is parallel to this triangle.

    float inv_det = 1.0f / det;
    vec4 s = VectorSubtract(ray_origin, triangle.p[0]);
    float u = inv_det * VectorDotProduct(s, ray_cross_e2);

    if (u < 0 || u > 1)
        return false;

    vec4 s_cross_e1 = VectorCrossProduct(s, edge1);
    float v = inv_det * VectorDotProduct(ray_vector, s_cross_e1);

    if (v < 0 || u + v > 1)
        return false;

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = inv_det * VectorDotProduct(edge2, s_cross_e1);

    if (t > epsilon) // ray intersection
    {
        vec4 temp = VectorMultiplication(ray_vector, t);
        intersection = VectorAddition(ray_origin, temp);

        return true;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return true;
}

__device__ vec4 Raycast(int x, int y, int screenWidth, int screenHeight, float fov, float AspectRatio)
{
    // Normalized Device Coordinates
    float xNDC = (2.0f * x) / screenWidth - 1.0f;
    float yNDC = (2.0f * y) / screenHeight - 1.0f ;
    float zNDC = 1.0f;

    vec4 normalized_device_coordinates_ray = { xNDC, yNDC, zNDC };

    // Homogeneous Clip Coordinates
    vec4 homogeneous_clip_coordinates_ray = { normalized_device_coordinates_ray.coo[0] * AspectRatio, normalized_device_coordinates_ray.coo[1] * AspectRatio * fov, 1.0f };

    return VectorNormalize(homogeneous_clip_coordinates_ray);
}

__host__ __device__ rgb RayTracing(vec4& normal, vec4& lightDirection, vec4& lightColor, vec4& ObjAlbedo) {
    vec4 lightningSend = RgbMultiplication(lightColor, ObjAlbedo);
    float theta = VectorDotProduct(normal, lightDirection);
    vec4 receivedIllumination = VectorMultiplication(lightningSend, theta);
    rgb finalColor = { receivedIllumination.coo[0] * 255.0f, receivedIllumination.coo[1] * 255.0f, receivedIllumination.coo[2] * 255.0f, 255 };
    return finalColor;
}

__global__ void rayTraceKernel(rgb* buffer, triangle* triangles, Light l, vec4 O_ray, int numTriangles, int screenWidth, int screenHeight, float fov, float AspectRatio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= screenWidth || y >= screenHeight) return;

    vec4 rayDir = Raycast(x, y, screenWidth, screenHeight, fov, AspectRatio);
    vec4 intersection;
    for (int i = 0; i < numTriangles; ++i) {
        if (ray_intersects_triangles(O_ray, rayDir, triangles[i], intersection)) {
            vec4 alb = { 1.0f, 1.0f, 1.0f };

            vec4 line1 = VectorSubtract(triangles[i].p[1], triangles[i].p[0]);
            vec4 line2 = VectorSubtract(triangles[i].p[2], triangles[i].p[0]);

            // It's normally normal to normalise the normal
            vec4 normal = VectorCrossProduct(line1, line2);
            normal = VectorNormalize(normal);
            vec4 lDir = VectorSubtract(l.lightPosition, intersection);
            lDir = VectorNormalize(lDir);
            buffer[y * screenWidth + x] = RayTracing(normal, lDir, l.lightColor, alb);
            break;
        }
    }
}


__global__ void clearBuffer(rgb* buffer, int screenWidth, int screenHeight, rgb defaultColor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < screenWidth && y < screenHeight) {
        buffer[y * screenWidth + x] = defaultColor;
    }
}

void drawBuffer(SDL_Renderer* renderer, std::vector<rgb>& bufferPixel, int ScreenSizeX, int ScreenSizeY) {
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, ScreenSizeX, ScreenSizeY);

    // Check if texture creation was successful
    if (!texture) {
        return;
    }

    // Create an array to store pixel data
    std::vector<Uint32> pixels(ScreenSizeX * ScreenSizeY);

    for (int y = 0; y < ScreenSizeY; ++y) {
        for (int x = 0; x < ScreenSizeX; ++x) {
            rgb& pixel = bufferPixel[y * ScreenSizeX + x];
            Uint8 r = static_cast<Uint8>(pixel.color[0]);
            Uint8 g = static_cast<Uint8>(pixel.color[1]);
            Uint8 b = static_cast<Uint8>(pixel.color[2]);
            Uint8 a = static_cast<Uint8>(pixel.color[3]);

            // Convert RGB color to Uint32
            pixels[y * ScreenSizeX + x] = (r << 24) | (g << 16) | (b << 8) | a;
        }
    }

    // Update texture with pixel data
    SDL_UpdateTexture(texture, nullptr, pixels.data(), ScreenSizeX * sizeof(Uint32));

    // Clear the renderer
    SDL_RenderClear(renderer);

    // Copy the texture to the renderer
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);

    // Present the renderer
    SDL_RenderPresent(renderer);

    // Destroy the texture
    SDL_DestroyTexture(texture);
}


int main(int argc, char* argv[]) {

    int ScreenSizeX = 0; int ScreenSizeY = 0;
    GetDesktopResolution(ScreenSizeX, ScreenSizeY);

    SDL_Window* win = NULL;
    SDL_Renderer* renderer = NULL;

    if (SDL_Init(SDL_INIT_VIDEO) < 0)
        return 1;

    win = SDL_CreateWindow("Game Engine 3D", 100, 100, ScreenSizeX, ScreenSizeY, SDL_WINDOW_SHOWN); // | SDL_WINDOW_FULLSCREEN_DESKTOP
    renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED); //  | SDL_RENDERER_PRESENTVSYNC
    
    float FovRad = 1.0f / tanf(90.0f * 0.5f * 3.1415926535f / 180.0f);
    float Far = 1000.0f;
    float Near = 0.1f;
    float AspectRatio = (ScreenSizeX) / (ScreenSizeY);
    float Theta = 0.0f;
    float Yaw = 0.0f;
    float elapsedTime = 0.0f;

    vec4 alb = { 1.0f, 1.0f, 1.0f, 1.0f };

    camera cam = { vec4({0.0f, 0.0f, 0.0f}), vec4({0.0f, 0.0f, 0.0f}), vec4({0.0f, 1.0f, 0.0f}), 0.2f};
    

    vec4 vLookSi = { 1.0f }; vec4 vLookFor = { 0.0f, 0.0f, 1.0f }; vec4 vSide = { 0.0f, 0.0f, 0.0f, 1.0f };

    std::string argv_str(argv[0]);
    std::string base = argv_str.substr(0, argv_str.find_last_of("\\"));
    std::string PathObject3D = base + "\\";
    std::string Mesh[] = { "teapot.obj" };
    //std::string MeshFileLoc[] = { "\\obj\\" };
    vec4 MeshCoo[] = { vec4({0.0f, 0.0f, 8.0f, 1.0f}) };
    vec4 MeshRot[] = { vec4({0.0, 0.0, 0.0f, 0.0f}) };

    meshes Meshes;
    /*for (int m = 0; m < (sizeof(Mesh) / sizeof(*Mesh)); m++)
    {

        mesh NewMesh;
        //Meshes.meshes[m].path = PathObject3D + MeshFileLoc[m];
        NewMesh.LoadFromObjectFile(PathObject3D + Mesh[m]);
        NewMesh.coo = MeshCoo[m];
        NewMesh.rot = MeshRot[m];
        NewMesh.Pos(MeshCoo[m].coo[0], MeshCoo[m].coo[1], MeshCoo[m].coo[2]);
        NewMesh.Scale(MeshCoo[m].coo[3]);
        Meshes.meshes.push_back(NewMesh);
    }*/

    mesh TestMesh = {{

            // SOUTH
            { 0.0f, 0.0f, 0.0f, 1.0f,    0.0f, 1.0f, 0.0f, 1.0f,    1.0f, 1.0f, 0.0f, 1.0f,		0.0f, 1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		1.0f, 0.0f, 1.0f,},
            { 0.0f, 0.0f, 0.0f, 1.0f,    1.0f, 1.0f, 0.0f, 1.0f,    1.0f, 0.0f, 0.0f, 1.0f,		0.0f, 1.0f, 1.0f,		1.0f, 0.0f, 1.0f,		1.0f, 1.0f, 1.0f,},

            // EAST           																			   
            { 1.0f, 0.0f, 0.0f, 1.0f,    1.0f, 1.0f, 0.0f, 1.0f,    1.0f, 1.0f, 1.0f, 1.0f,		0.0f, 1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		1.0f, 0.0f, 1.0f,},
            { 1.0f, 0.0f, 0.0f, 1.0f,    1.0f, 1.0f, 1.0f, 1.0f,    1.0f, 0.0f, 1.0f, 1.0f,		0.0f, 1.0f, 1.0f,		1.0f, 0.0f, 1.0f,		1.0f, 1.0f, 1.0f,},

            // NORTH           																			   
            { 1.0f, 0.0f, 1.0f, 1.0f,    1.0f, 1.0f, 1.0f, 1.0f,    0.0f, 1.0f, 1.0f, 1.0f,		0.0f, 1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		1.0f, 0.0f, 1.0f,},
            { 1.0f, 0.0f, 1.0f, 1.0f,    0.0f, 1.0f, 1.0f, 1.0f,    0.0f, 0.0f, 1.0f, 1.0f,		0.0f, 1.0f, 1.0f,		1.0f, 0.0f, 1.0f,		1.0f, 1.0f, 1.0f,},

            // WEST            																			   
            { 0.0f, 0.0f, 1.0f, 1.0f,    0.0f, 1.0f, 1.0f, 1.0f,    0.0f, 1.0f, 0.0f, 1.0f,		0.0f, 1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		1.0f, 0.0f, 1.0f,},
            { 0.0f, 0.0f, 1.0f, 1.0f,    0.0f, 1.0f, 0.0f, 1.0f,    0.0f, 0.0f, 0.0f, 1.0f,		0.0f, 1.0f, 1.0f,		1.0f, 0.0f, 1.0f,		1.0f, 1.0f, 1.0f,},

            // TOP             																			   
            { 0.0f, 1.0f, 0.0f, 1.0f,    0.0f, 1.0f, 1.0f, 1.0f,    1.0f, 1.0f, 1.0f, 1.0f,		0.0f, 1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		1.0f, 0.0f, 1.0f,},
            { 0.0f, 1.0f, 0.0f, 1.0f,    1.0f, 1.0f, 1.0f, 1.0f,    1.0f, 1.0f, 0.0f, 1.0f,		0.0f, 1.0f, 1.0f,		1.0f, 0.0f, 1.0f,		1.0f, 1.0f, 1.0f,},

            // BOTTOM          																			  
            { 1.0f, 0.0f, 1.0f, 1.0f,    0.0f, 0.0f, 1.0f, 1.0f,    0.0f, 0.0f, 0.0f, 1.0f,		0.0f, 1.0f, 1.0f,		0.0f, 0.0f, 1.0f,		1.0f, 0.0f, 1.0f,},
            { 1.0f, 0.0f, 1.0f, 1.0f,    0.0f, 0.0f, 0.0f, 1.0f,    1.0f, 0.0f, 0.0f, 1.0f,		0.0f, 1.0f, 1.0f,		1.0f, 0.0f, 1.0f,		1.0f, 1.0f, 1.0f,},

        } };
    TestMesh.Pos(0.0f, 0.0f, 10.0f);
    Meshes.meshes.push_back(TestMesh);

    std::vector<triangle> combined;

    for (int s = 0; s < Meshes.meshes.size(); s++)
    {
        for (int t = 0; t < Meshes.meshes[s].tris.size(); t++)
        {
            combined.push_back(Meshes.meshes[s].tris[t]);
        }
    }

    Uint32 startTime = SDL_GetTicks();
    int frameCount = 0;
    float fps = 0.0f;
    bool running = true;
    bool keys[SDL_NUM_SCANCODES] = { false };
    while (running) {
        Light light = { cam.cameraPosition, vec4({ 1.0f, 1.0f, 1.0f, 1.0f}), 1.0 };
        SDL_Event event;
        
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT) { running = false; break; }
            else if (event.type == SDL_KEYDOWN) { keys[event.key.keysym.scancode] = true; } // User presses a key
            else if (event.type == SDL_KEYUP) { keys[event.key.keysym.scancode] = false; } // User releases a key
        }

        vec4 vSide = VectorMultiplication(vLookSi, cam.velocity);
        cam.cameraLookAt = VectorMultiplication(vLookFor, cam.velocity);

        if (keys[SDL_SCANCODE_ESCAPE]) { running = false; }
        if (keys[SDL_SCANCODE_UP]) { cam.cameraPosition.coo[1] -= cam.velocity; }
        if (keys[SDL_SCANCODE_DOWN]) { cam.cameraPosition.coo[1] += cam.velocity; }
        if (keys[SDL_SCANCODE_LEFT]) { cam.cameraPosition = VectorSubtract(cam.cameraPosition, vSide); }
        if (keys[SDL_SCANCODE_RIGHT]) { cam.cameraPosition = VectorAddition(cam.cameraPosition, vSide); }
        if (keys[SDL_SCANCODE_W]) { cam.cameraPosition = VectorAddition(cam.cameraPosition, cam.cameraLookAt); }
        if (keys[SDL_SCANCODE_A]) { Yaw += cam.velocity; }
        if (keys[SDL_SCANCODE_S]) { cam.cameraPosition = VectorSubtract(cam.cameraPosition, cam.cameraLookAt); }
        if (keys[SDL_SCANCODE_D]) { Yaw -= cam.velocity; }

        // Allocate device memory
        rgb* d_buffer;
        triangle* d_triangles;
        cudaMalloc(&d_buffer, ScreenSizeY * ScreenSizeX * sizeof(rgb));
        cudaMalloc(&d_triangles, combined.size() * sizeof(triangle));

        // Copy data to device
        cudaMemcpy(d_triangles, combined.data(), combined.size() * sizeof(triangle), cudaMemcpyHostToDevice);

        // Define block and grid sizes
        dim3 block(32, 32);
        dim3 grid;
        grid.x = (ScreenSizeX + block.x - 1) / block.x;
        grid.y = (ScreenSizeY + block.y - 1) / block.y;

        // Clear the buffer
        rgb defaultColor = { 0.0f, 0.0f, 255.0f, 255.0f };
        clearBuffer << <grid, block >> > (d_buffer, ScreenSizeX, ScreenSizeY, defaultColor);
        cudaDeviceSynchronize();
        //double buffer
        // Launch the ray tracing kernel
        rayTraceKernel << <grid, block >> > (d_buffer, d_triangles, light, cam.cameraPosition, combined.size(), ScreenSizeX, ScreenSizeY, FovRad, AspectRatio);
        cudaDeviceSynchronize();

        // Copy the result back to host
        std::vector<rgb> bufferPixel(ScreenSizeY * ScreenSizeX);
        cudaMemcpy(bufferPixel.data(), d_buffer, ScreenSizeY * ScreenSizeX * sizeof(rgb), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_buffer);
        cudaFree(d_triangles);

        drawBuffer(renderer, bufferPixel, ScreenSizeX, ScreenSizeY);

        // FPS calculation
        frameCount++;
        Uint32 frameEnd = SDL_GetTicks();
        Uint32 elapsedTime = frameEnd - startTime;
        fps = frameCount / (elapsedTime / 1000.0f);
        if (elapsedTime > 1000) { frameCount = 0; startTime = frameEnd; } // Update every second
        std::string title = "GameEngine {Info : { FPS: " + std::to_string(fps) + ", Number of Triangles: " + std::to_string(combined.size()) + " }}";
        SDL_SetWindowTitle(win, title.c_str());
    }
    return 0;
}