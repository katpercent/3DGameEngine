#include<map>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<sstream>
#include<algorithm>
#include <stdio.h>
#include <SDL.h>
#include <cmath>

struct vec4
{
    float coo[4] = { 0.0, 0.0, 0.0, 1.0 };
};

struct vec3
{
    float u = 0;
    float v = 0;
    float w = 1;
};

struct triangle
{
    vec4 p[3];
    SDL_Color col[3];
};

// Definition of mesh structure
struct mesh {
    std::string path;
    std::vector<triangle> tris;
    vec4 coo;
    vec4 rot;

    bool LoadFromObjectFile(const std::string& sFilename) {
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
            else if (line[0] == 'f' && line[1] == ' ') {
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
        }

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

vec4 VectorSubtract(vec4& a, vec4& b)
{
    vec4 result;
    result.coo[0] = a.coo[0] - b.coo[0];
    result.coo[1] = a.coo[1] - b.coo[1];
    result.coo[2] = a.coo[2] - b.coo[2];
    return result;
};

vec4 VectorAddition(vec4& a, vec4& b)
{
    vec4 result;
    result.coo[0] = a.coo[0] + b.coo[0];
    result.coo[1] = a.coo[1] + b.coo[1];
    result.coo[2] = a.coo[2] + b.coo[2];
    return result;
};

vec4 VectorMultiplication(vec4& a, float& k)
{
    vec4 result;
    result.coo[0] = a.coo[0] * k;
    result.coo[1] = a.coo[1] * k;
    result.coo[2] = a.coo[2] * k;
    return result;
};

vec4 VectorDivision(vec4& a, float& k)
{
    vec4 result;
    if (k != 0) {
        result.coo[0] = a.coo[0] / k;
        result.coo[1] = a.coo[1] / k;
        result.coo[2] = a.coo[2] / k;
    }
    return result;
};

vec4 VectorMultiplyVector(vec4& a, vec4& b)
{
    vec4 result;
    result.coo[0] = a.coo[0] * b.coo[0] + a.coo[0] * b.coo[1] + a.coo[0] * b.coo[2];
    result.coo[1] = a.coo[1] * b.coo[0] + a.coo[1] * b.coo[1] + a.coo[1] * b.coo[2];
    result.coo[2] = a.coo[2] * b.coo[0] + a.coo[2] * b.coo[1] + a.coo[2] * b.coo[2];
    return result;
};

float VectorDotProduct(vec4 v1, vec4 v2)
{
    float result = v1.coo[0] * v2.coo[0] + v1.coo[1] * v2.coo[1] + v1.coo[2] * v2.coo[2];
    return result;
};

float VectorLength(vec4& v)
{
    return sqrtf(VectorDotProduct(v, v));
};

vec4 VectorNormalise(vec4& v)
{
    float l = VectorLength(v);
    if (l != 0) {
        v.coo[0] = v.coo[0] / l; v.coo[1] = v.coo[1] / l; v.coo[2] = v.coo[2] / l;
    }
    return v;
};

vec4 VectorCrossProduct(vec4& v1, vec4& v2)
{
    vec4 result;
    result.coo[0] = v1.coo[1] * v2.coo[2] - v1.coo[2] * v2.coo[1];
    result.coo[1] = v1.coo[2] * v2.coo[0] - v1.coo[0] * v2.coo[2];
    result.coo[2] = v1.coo[0] * v2.coo[1] - v1.coo[1] * v2.coo[0];
    return result;
};

vec4 VectorInverse(vec4& v)
{
    v.coo[0] *= -1; v.coo[1] *= -1; v.coo[2] *= -1;
    return v;
};

mat4x4 MatrixMakeIdentity()
{
    mat4x4 matrix;
    matrix.m[0][0] = 1.0;
    matrix.m[1][1] = 1.0;
    matrix.m[2][2] = 1.0;
    matrix.m[3][3] = 1.0;
    return matrix;
};

void MatrixMultiplyVector(vec4& o, mat4x4& m, vec4& i)
{
    o.coo[0] = i.coo[0] * m.m[0][0] + i.coo[1] * m.m[0][1] + i.coo[2] * m.m[0][2] + m.m[0][3];
    o.coo[1] = i.coo[0] * m.m[1][0] + i.coo[1] * m.m[1][1] + i.coo[2] * m.m[1][2] + m.m[1][3];
    o.coo[2] = i.coo[0] * m.m[2][0] + i.coo[1] * m.m[2][1] + i.coo[2] * m.m[2][2] + m.m[2][3];
    o.coo[3] = i.coo[0] * m.m[3][0] + i.coo[1] * m.m[3][1] + i.coo[2] * m.m[3][2] + m.m[3][3];

};

mat4x4 MatrixMultiplyMatrix(mat4x4& m1, mat4x4& m2)
{
    mat4x4 matrix;
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            matrix.m[r][c] = m1.m[r][0] * m2.m[0][c] + m1.m[r][1] * m2.m[1][c] + m1.m[r][2] * m2.m[2][c] + m1.m[r][3] * m2.m[3][c];
    return matrix;
}

mat4x4 MatrixTranslation(vec4 coordinates)
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

mat4x4 MatrixPointAt(vec4& pos, vec4& target, vec4& up)
{

    /* [[newRightx, newUpx, newForwardx, 0.0]
    [newRighty, newUpy, newForwardy, 0.0]
    [newRightz, newUpz, newForwardz, 0.0]
    [dp1, dp2, dp3, 0.0]]*/

    vec4 newForward = VectorSubtract(target, pos);
    VectorNormalise(newForward);

    float r = VectorDotProduct(up, newForward);
    vec4 a = VectorMultiplication(newForward, r);
    vec4 newUp = VectorSubtract(up, a);
    VectorNormalise(newUp);

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

mat4x4 MatrixRotationX(float& AngleRad)
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

mat4x4 MatrixRotationY(float& AngleRad)
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

mat4x4 MatrixRotationZ(float& AngleRad)
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

vec4 Vector_IntersectPlane(vec4& plane_p, vec4& plane_n, vec4& lineStart, vec4& lineEnd, float& t)
{
    VectorNormalise(plane_n);
    float plane_d = -VectorDotProduct(plane_n, plane_p);
    float ad = VectorDotProduct(lineStart, plane_n);
    float bd = VectorDotProduct(lineEnd, plane_n);
    t = (-plane_d - ad) / (bd - ad);
    vec4 lineStartToEnd = VectorSubtract(lineEnd, lineStart);
    vec4 lineToIntersect = VectorMultiplication(lineStartToEnd, t);
    return VectorAddition(lineStart, lineToIntersect);;
}

int Triangle_ClipAgainstPlane(vec4 plane_p, vec4 plane_n, triangle& in_tri, triangle& out_tri1, triangle& out_tri2)
{
    // Make sure plane normal is indeed normal
    VectorNormalise(plane_n);

    // Return signed shortest distance from point to plane
    auto dist = [&](vec4& p)
        {
            return (plane_n.coo[0] * p.coo[0] + plane_n.coo[1] * p.coo[1] + plane_n.coo[2] * p.coo[2] - VectorDotProduct(plane_n, plane_p));
        };

    // Create two temporary storage arrays to classify points either side of plane
    // If distance sign is positive, point lies on "inside" of plane
    vec4* inside_points[3];  int nInsidePointCount = 0;
    vec4* outside_points[3]; int nOutsidePointCount = 0;

    // Get signed distance of each point in triangle to plane
    float d0 = dist(in_tri.p[0]);
    float d1 = dist(in_tri.p[1]);
    float d2 = dist(in_tri.p[2]);

    if (d0 >= 0) { inside_points[nInsidePointCount++] = &in_tri.p[0]; }
    else {
        outside_points[nOutsidePointCount++] = &in_tri.p[0];
    }
    if (d1 >= 0) {
        inside_points[nInsidePointCount++] = &in_tri.p[1];
    }
    else {
        outside_points[nOutsidePointCount++] = &in_tri.p[1];
    }
    if (d2 >= 0) {
        inside_points[nInsidePointCount++] = &in_tri.p[2];
    }
    else {
        outside_points[nOutsidePointCount++] = &in_tri.p[2];
    }

    // Now classify triangle points, and break the input triangle into 
    // smaller output triangles if required. There are four possible
    // outcomes...

    if (nInsidePointCount == 0)
    {
        // All points lie on the outside of plane, so clip whole triangle
        // It ceases to exist

        return 0; // No returned triangles are valid
    }

    if (nInsidePointCount == 3)
    {
        // All points lie on the inside of plane, so do nothing
        // and allow the triangle to simply pass through
        out_tri1 = in_tri;

        return 1; // Just the one returned original triangle is valid
    }

    if (nInsidePointCount == 1 && nOutsidePointCount == 2)
    {
        // Triangle should be clipped. As two points lie outside
        // the plane, the triangle simply becomes a smaller triangle

        // Copy appearance info to new triangle
        for (int i = 0; i < 3; ++i)
            out_tri1.col[i] = in_tri.col[i];

        // The inside point is valid, so keep that...
        out_tri1.p[0] = *inside_points[0];

        // but the two new points are at the locations where the 
        // original sides of the triangle (lines) intersect with the plane
        float t;
        out_tri1.p[1] = Vector_IntersectPlane(plane_p, plane_n, *inside_points[0], *outside_points[0], t);

        out_tri1.p[2] = Vector_IntersectPlane(plane_p, plane_n, *inside_points[0], *outside_points[1], t);

        return 1; // Return the newly formed single triangle
    }

    if (nInsidePointCount == 2 && nOutsidePointCount == 1)
    {
        // Triangle should be clipped. As two points lie inside the plane,
        // the clipped triangle becomes a "quad". Fortunately, we can
        // represent a quad with two new triangle

        // The first triangle consists of the two inside points and a new
        // point determined by the location where one side of the triangle
        // intersects with the plane
        out_tri1.p[0] = *inside_points[0];
        out_tri1.p[1] = *inside_points[1];

        float t;
        out_tri1.p[2] = Vector_IntersectPlane(plane_p, plane_n, *inside_points[0], *outside_points[0], t);

        // The second triangle is composed of one of the inside points, a
        // new point determined by the intersection of the other side of the 
        // triangle and the plane, and the newly created point above
        out_tri2.p[0] = *inside_points[1];
        out_tri2.p[1] = out_tri1.p[2];
        out_tri2.p[2] = Vector_IntersectPlane(plane_p, plane_n, *inside_points[1], *outside_points[0], t);

        // Copy appearance info to new triangles
        for (int i = 0; i < 3; ++i)
        {
            out_tri1.col[i] = in_tri.col[i];
            out_tri2.col[i] = in_tri.col[i];
        }

        return 2; // Return two newly formed triangles which form a quad
    }
}

int maxValue(int a, int b) {
    if (a > b)
        return a;
    else return b;
};

int minValue(int a, int b) {
    if (a > b)
        return b;
    else return a;
};

double degreesToRadians(double degrees) {
    return degrees * (M_PI / 180.0);
}

// Convert time (in hours) to radians representing the sun's position
double timeToRadians(double& time) {
    // Convert time to fraction of a day (0 to 1)
    double fractionOfDay = fmod(time / 24.0, 1.0);
    // Convert fraction of a day to radians (0 to 2π)
    return fractionOfDay * 2 * M_PI;
}

vec4 soleil(vec4& vec, double& time)
{
    float radians = timeToRadians(time);
    mat4x4 matRot = MatrixRotationZ(radians);
    MatrixMultiplyVector(vec, matRot, vec);
    return vec;
}

static SDL_Color PhongLighting(vec4& lightColor, vec4& normal, vec4& lightDir, vec4& viewDir, float ambientStrength, float diffuseStrength, float specularStrength, float shininess)
{
    // Ambient component
    vec4 ambient = VectorMultiplication(lightColor, ambientStrength);

    // Diffuse component
    float diff = maxValue(0.0f, VectorDotProduct(normal, lightDir));
    vec4 diffuse = VectorMultiplication(lightColor, diff);

    // Specular component
    float para2 = 2.0f * VectorDotProduct(normal, lightDir);
    vec4 para = VectorMultiplication(normal, para2);
    vec4 reflectDir = VectorSubtract(para, lightDir);
    float spec = pow(maxValue(0.0f, VectorDotProduct(viewDir, VectorNormalise(reflectDir))), shininess);
    float para3 = spec * specularStrength;
    vec4 specular = VectorMultiplication(lightColor, para3);

    // Combine all components
    para = VectorAddition(lightColor, diffuse);
    vec4 intensity = VectorAddition(para, specular);
    float rgbLimit = 255;
    vec4 lighting = VectorMultiplication(intensity, rgbLimit);
    SDL_Color colour = { lighting.coo[0], lighting.coo[1], lighting.coo[2], 255};
    return colour;
}

// Oren-Nayar reflection model function
SDL_Color orenNayar(vec4& normal, vec4& lightDir, vec4& viewDir, float roughness) {
    float sigma2 = roughness * roughness;

    // Calculate angles
    float theta_r = std::acos(VectorDotProduct(normal, viewDir));
    float theta_i = std::acos(VectorDotProduct(normal, lightDir));

    // Calculate the phi term
    float p2 = VectorDotProduct(normal, lightDir); vec4 p1 = VectorMultiplication(normal, p2);
    vec4 lightPerpendicular = VectorSubtract(lightDir, p1);
    float p4 = VectorDotProduct(normal, viewDir); vec4 p3 = VectorMultiplication(normal, p4);
    vec4 viewPerpendicular = VectorSubtract(viewDir, p3);
    float cos_phi_diff = VectorDotProduct(lightPerpendicular, viewPerpendicular) / (VectorLength(lightPerpendicular) * VectorLength(viewPerpendicular));

    // Oren-Nayar coefficients
    float A = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));
    float B = 0.45f * sigma2 / (sigma2 + 0.09f);

    // Calculate final intensity
    float alpha = maxValue(theta_r, theta_i);
    float beta = minValue(theta_r, theta_i);

    Uint8 intensity = A + B * maxValue(0.0f, cos_phi_diff) * std::sin(alpha) * std::tan(beta);
    intensity = intensity * maxValue(0.0f, VectorDotProduct(normal, lightDir));
    // Scale by the Lambertian cosine term
    return SDL_Color{ intensity, intensity, intensity, 255 };
}

SDL_Color Gouraud(vec4& normal, vec4& lightDir, vec4& lightColor) {
    float intensity = VectorDotProduct(normal, lightDir);
    intensity = fmax(0.0, intensity); // Ensure intensity is non-negative
    vec4 color = VectorMultiplication(lightColor, intensity);
    SDL_Color finalColor = { color.coo[0], color.coo[1], color.coo[2], 255 };
    return finalColor;
}


int main(int argc, char* argv[])
{
    SDL_Window* win = NULL;
    SDL_Renderer* renderer = NULL;
    SDL_Texture* img = NULL;

    if (SDL_Init(SDL_INIT_VIDEO) < 0)
        return 1;

    SDL_DisplayMode DM;
    SDL_GetCurrentDisplayMode(0, &DM);
    int ScreenSizeX = DM.w - 100;
    int ScreenSizeY = DM.h - 100;

    int StartHour = 6;
    int TickRate = 10;
    float Velocity = 0.4;
    float VelocityCam = 0.2;
    float FovRad = 1.0f / tanf(90.0f * 0.5f * 3.1415926535f / 180.0f);
    float Far = 1000.0f;
    float Near = 0.1f;
    float AspectRatio = (ScreenSizeX) / (ScreenSizeY);
    float Theta = 0.0f;
    float Yaw = 0.0f;
    float elapsedTime = 0.0f;
    
    vec4 vCamera; vec4 vLookSi = { 1.0 }; vec4 vLookFor = { 0.0, 0.0, 1.0 }; vec4 vSide; vec4 vFor;
    mat4x4 MatrixProjection;
    MatrixProjection.m[0][0] = AspectRatio * FovRad;
    MatrixProjection.m[1][1] = FovRad;
    MatrixProjection.m[2][2] = Far / (Far - Near);
    MatrixProjection.m[2][3] = (-Far * Near) / (Far - Near);
    MatrixProjection.m[3][2] = 1.0f;


    std::string argv_str(argv[0]);
    std::string base = argv_str.substr(0, argv_str.find_last_of("\\"));
    std::string PathObject3D = base + "\\obj\\";
    vec4 lightPos = { 0.0f, 0.0f, 0.0f, 1.0f};
    std::string Mesh[] = { "Wall.obj" };
    vec4 MeshCoo[] = { vec4({0.0f, 0.0f, 20.0f, 0.04f}) };
    vec4 MeshRot[] = { vec4({3.1415926535f / 2.0f, 2.0f * 3.1415926535f / 2.0f, 0.0f, 0.0f}) };
    // 3.1415926535f/2.0f
    
    meshes Meshes;
    for (int m = 0; m < (sizeof(Mesh) / sizeof(*Mesh)); m++)
    {

        mesh NewMesh;
        Meshes.meshes.push_back(NewMesh);
        Meshes.meshes[m].path = PathObject3D + Mesh[m];
        Meshes.meshes[m].LoadFromObjectFile(PathObject3D + Mesh[m]);
        Meshes.meshes[m].coo = MeshCoo[m];
        Meshes.meshes[m].rot = MeshRot[m];
        Meshes.meshes[m].Scale(MeshCoo[m].coo[3]);
    }

    win = SDL_CreateWindow("Game Engine 3D", 100, 100, ScreenSizeX, ScreenSizeY, SDL_WINDOW_SHOWN); // | SDL_WINDOW_FULLSCREEN_DESKTOP
    renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED); //  | SDL_RENDERER_PRESENTVSYNC

    Uint32 startTime = SDL_GetTicks();

    int frameCount = 0;
    float fps = 0.0f;

    bool running = true;
    while (running)
    {
        int numVert = 0;
        SDL_SetRenderDrawColor(renderer, 20, 40, 100, 255);
        SDL_RenderClear(renderer);
        SDL_Event event;

        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                running = false;
                break;
            }
            // User presses a key
            else if (event.type == SDL_KEYDOWN)
            {
                vec4 vSide = VectorMultiplication(vLookSi, Velocity);
                vec4 vForward = VectorMultiplication(vLookFor, Velocity);

                if (SDLK_ESCAPE == event.key.keysym.sym) { running = false; break; }
                if (SDLK_UP == event.key.keysym.sym) { vCamera.coo[1] += Velocity; break; }
                if (SDLK_DOWN == event.key.keysym.sym) { vCamera.coo[1] -= Velocity; break; }
                if (SDLK_LEFT == event.key.keysym.sym) { vCamera = VectorAddition(vCamera, vSide); break; }
                if (SDLK_RIGHT == event.key.keysym.sym) { vCamera = VectorSubtract(vCamera, vSide); break; }
                if (SDLK_w == event.key.keysym.sym) { vCamera = VectorAddition(vCamera, vForward); break; }
                if (SDLK_a == event.key.keysym.sym) { Yaw += VelocityCam; break; }
                if (SDLK_s == event.key.keysym.sym) { vCamera = VectorSubtract(vCamera, vForward); break; }
                if (SDLK_d == event.key.keysym.sym) { Yaw -= VelocityCam; break; }
            }
        }
        
        mat4x4 matRotX = MatrixRotationX(Theta);
        mat4x4 matRotY = MatrixRotationY(Theta);
        mat4x4 matRotZ = MatrixRotationZ(Theta);
        
        mat4x4 matWorld = MatrixMakeIdentity();
        matWorld = MatrixMultiplyMatrix(matRotZ, matRotX);
        //matWorld = MatrixMultiplyMatrix(matWorld, matRotY);

        vec4 vUp = { 0.0, 1.0, 0.0 };
        vec4 vTargetZ = { 0.0, 0.0, 1.0 };
        vec4 vTargetX = { 1.0, 0.0, 0.0 };

        mat4x4 matCameraRot = MatrixRotationY(Yaw);
        MatrixMultiplyVector(vLookFor, matCameraRot, vTargetZ);
        MatrixMultiplyVector(vLookSi, matCameraRot, vTargetX);
        vTargetZ = VectorAddition(vCamera, vLookFor);
        mat4x4 matView = MatrixPointAt(vCamera, vTargetZ, vUp);

        std::vector<triangle> vecTrianglesToSort;
        for (auto& mesh : Meshes.meshes)
        {
            mat4x4 matMeshRotX = MatrixRotationX(mesh.rot.coo[0]);
            mat4x4 matMeshRotY = MatrixRotationY(mesh.rot.coo[1]);
            mat4x4 matMeshRotZ = MatrixRotationZ(mesh.rot.coo[2]);
            for (auto& tri : mesh.tris)
            {
                triangle triProjected, triViewed, triTranslated, triTransformed, triRotatedX, triRotatedXY, triRotatedXYZ;

                for (int o = 0; o < 3; o++)
                    MatrixMultiplyVector(triRotatedX.p[o], matMeshRotX, tri.p[o]);
                for (int o = 0; o < 3; o++)
                     MatrixMultiplyVector(triRotatedXY.p[o], matMeshRotY, triRotatedX.p[o]);
                for (int o = 0; o < 3; o++)
                    MatrixMultiplyVector(triRotatedXYZ.p[o], matMeshRotZ, triRotatedXY.p[o]);

                MatrixMultiplyVector(triTransformed.p[0], matWorld, triRotatedXYZ.p[0]);
                MatrixMultiplyVector(triTransformed.p[1], matWorld, triRotatedXYZ.p[1]);
                MatrixMultiplyVector(triTransformed.p[2], matWorld, triRotatedXYZ.p[2]);

                for (int i = 0; i < 3; i++) {
                    triTranslated.p[i].coo[0] = triTransformed.p[i].coo[0] + mesh.coo.coo[0];
                    triTranslated.p[i].coo[1] = triTransformed.p[i].coo[1] + mesh.coo.coo[1];
                    triTranslated.p[i].coo[2] = triTransformed.p[i].coo[2] + mesh.coo.coo[2];
                }

                vec4 line1 = VectorSubtract(triTranslated.p[1], triTranslated.p[0]);
                vec4 line2 = VectorSubtract(triTranslated.p[2], triTranslated.p[0]);

                // It's normally normal to normalise the normal
                vec4 normal = VectorCrossProduct(line1, line2);
                VectorNormalise(normal);

                vec4 vCameraRay = VectorSubtract(triTranslated.p[0], vCamera);

                //if (normal.coo[2] < 0)
                if (VectorDotProduct(normal, vCameraRay) < 0.0f)
                {
                    for (int i = 0; i < 3; i++)
                    {
                        
                        vec4 lightDir = VectorSubtract(lightPos, triProjected.p[i]); // Assuming lightPos is the position of the light source
                        VectorNormalise(lightDir);
                        vec4 viewDir = VectorSubtract(vCamera, triProjected.p[i]); // Assuming cameraPos is the position of the camera
                        VectorNormalise(viewDir);

                        // Calculate Phong lighting
                        //vec4& normal, vec4& lightDir, vec4& viewDir, float ambientStrength, float diffuseStrength, float specularStrength, float shininess
                        vec4 lightColor = { 1.0f, 0.0f, 0.0f, 0.0f };
                        //triTranslated.col[i] = Gouraud(normal, lightDir, lightColor);
                        //triTranslated.col[i] = orenNayar(normal, lightDir, viewDir, 0.0f); // Adjust the parameters as needed
                        triTranslated.col[i] = PhongLighting(lightColor, normal, lightDir, viewDir, 0.7f, 0.6f, 0.5f, 256.0f); // Adjust the parameters as needed
                    }

                    // Convert World Space-- > View Space
                    MatrixMultiplyVector(triViewed.p[0], matView, triTranslated.p[0]);
                    MatrixMultiplyVector(triViewed.p[1], matView, triTranslated.p[1]);
                    MatrixMultiplyVector(triViewed.p[2], matView, triTranslated.p[2]);

                    int nClippedTriangles = 0;
                    triangle clipped[2];
                    nClippedTriangles = Triangle_ClipAgainstPlane({ 0.0f, 0.0f, 0.1f }, { 0.0f, 0.0f, 1.0f }, triViewed, clipped[0], clipped[1]);

                    // We may end up with multiple triangles form the clip, so project as
                    // required
                    for (int n = 0; n < nClippedTriangles; n++)
                    {
                        // Project triangles from 3D --> 2D
                        MatrixMultiplyVector(triProjected.p[0], MatrixProjection, clipped[n].p[0]);
                        MatrixMultiplyVector(triProjected.p[1], MatrixProjection, clipped[n].p[1]);
                        MatrixMultiplyVector(triProjected.p[2], MatrixProjection, clipped[n].p[2]);

                        triProjected.p[0] = VectorDivision(triProjected.p[0], triProjected.p[0].coo[3]);
                        triProjected.p[1] = VectorDivision(triProjected.p[1], triProjected.p[1].coo[3]);
                        triProjected.p[2] = VectorDivision(triProjected.p[2], triProjected.p[2].coo[3]);

                        // X/Y are inverted so put them back
                        triProjected.p[0].coo[0] *= -1.0f;
                        triProjected.p[1].coo[0] *= -1.0f;
                        triProjected.p[2].coo[0] *= -1.0f;
                        triProjected.p[0].coo[1] *= -1.0f;
                        triProjected.p[1].coo[1] *= -1.0f;
                        triProjected.p[2].coo[1] *= -1.0f;

                        // Scale into view
                        vec4 vOffsetView = { 1.0, 1.0, 0.0 };
                        triProjected.p[0] = VectorAddition(triProjected.p[0], vOffsetView);
                        triProjected.p[1] = VectorAddition(triProjected.p[1], vOffsetView);
                        triProjected.p[2] = VectorAddition(triProjected.p[2], vOffsetView);
                        triProjected.p[0].coo[0] *= 0.5f * ScreenSizeX;
                        triProjected.p[0].coo[1] *= 0.5f * ScreenSizeY;
                        triProjected.p[1].coo[0] *= 0.5f * ScreenSizeX;
                        triProjected.p[1].coo[1] *= 0.5f * ScreenSizeY;
                        triProjected.p[2].coo[0] *= 0.5f * ScreenSizeX;
                        triProjected.p[2].coo[1] *= 0.5f * ScreenSizeY;
                        for (int x = 0; x < 3; x++)
                        {
                            triProjected.col[x] = triTranslated.col[x];
                        }
                        // Store triangle for sorting
                        vecTrianglesToSort.push_back(triProjected);
                    }
                }
            }
        }


        // Sort triangles from back to front
        std::sort(vecTrianglesToSort.begin(), vecTrianglesToSort.end(), [](triangle& t1, triangle& t2)
        {
            float z1 = (t1.p[0].coo[2] + t1.p[1].coo[2] + t1.p[2].coo[2]) / 3.0f;
            float z2 = (t2.p[0].coo[2] + t2.p[1].coo[2] + t2.p[2].coo[2]) / 3.0f;
            return z1 > z2;
        });

        // Clip triangles against all four screen edges
        int triDraw = 0;
        // Clip triangles against all four screen edges
        for (auto& tri : vecTrianglesToSort) {
            std::vector<triangle> clippedTriangles = { tri };
            std::vector<triangle> trianglesToAdd;

            for (int plane = 0; plane < 4; ++plane) {
                trianglesToAdd.clear();
                for (auto& t : clippedTriangles) {
                    int nClippedTriangles;
                    triangle clipped[2]; // To store up to 2 clipped triangles

                    switch (plane) {
                    case 0: // Clip against left edge
                        nClippedTriangles = Triangle_ClipAgainstPlane(vec4{ 0.0f, 0.0f, 0.0f }, vec4{ 1.0f, 0.0f, 0.0f }, t, clipped[0], clipped[1]);
                        break;
                    case 1: // Clip against right edge
                        nClippedTriangles = Triangle_ClipAgainstPlane(vec4{ (float)(ScreenSizeX - 1), 0.0f, 0.0f }, vec4{ -1.0f, 0.0f, 0.0f }, t, clipped[0], clipped[1]);
                        break;
                    case 2: // Clip against top edge
                        nClippedTriangles = Triangle_ClipAgainstPlane(vec4{ 0.0f, 0.0f, 0.0f }, vec4{ 0.0f, 1.0f, 0.0f }, t, clipped[0], clipped[1]);
                        break;
                    case 3: // Clip against bottom edge
                        nClippedTriangles = Triangle_ClipAgainstPlane(vec4{ 0.0f, (float)(ScreenSizeY - 1), 0.0f }, vec4{ 0.0f, -1.0f, 0.0f }, t, clipped[0], clipped[1]);
                        break;
                    }

                    for (int i = 0; i < nClippedTriangles; ++i) {
                        trianglesToAdd.push_back(clipped[i]);
                    }
                }
                clippedTriangles = trianglesToAdd;
            }
        } 

        for (auto& triProjected : vecTrianglesToSort)
        {
            const std::vector<SDL_Vertex> verts =
            {
                { SDL_FPoint{ triProjected.p[0].coo[0], triProjected.p[0].coo[1] }, triProjected.col[0], SDL_FPoint{0} },
                { SDL_FPoint{ triProjected.p[1].coo[0], triProjected.p[1].coo[1] }, triProjected.col[1], SDL_FPoint{0} },
                { SDL_FPoint{ triProjected.p[2].coo[0], triProjected.p[2].coo[1] }, triProjected.col[2], SDL_FPoint{0} },
            };

            /*SDL_RenderDrawLine(renderer, triProjected.p[0].coo[0], triProjected.p[0].coo[1], triProjected.p[1].coo[0], triProjected.p[1].coo[1]);
            SDL_RenderDrawLine(renderer, triProjected.p[1].coo[0], triProjected.p[1].coo[1], triProjected.p[2].coo[0], triProjected.p[2].coo[1]);
            SDL_RenderDrawLine(renderer, triProjected.p[2].coo[0], triProjected.p[2].coo[1], triProjected.p[0].coo[0], triProjected.p[0].coo[1]);*/
            SDL_RenderGeometry(renderer, nullptr, verts.data(), verts.size(), nullptr, 0);
            numVert += 1;
        }

        // FPS calculation
        frameCount++;
        Uint32 frameEnd = SDL_GetTicks();
        Uint32 elapsedTime = frameEnd - startTime;
        if (elapsedTime > 1000) { // Update every second
            fps = frameCount / (elapsedTime / 1000.0f);
            frameCount = 0;
            startTime = frameEnd;
            std::string title = "GameEngine {Info : {FPS: " + std::to_string(fps) + ", Number of Triangles: " + std::to_string(numVert) + "}}";
            SDL_SetWindowTitle(win, title.c_str());
        }

        SDL_RenderPresent(renderer);
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
};