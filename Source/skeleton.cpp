#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModelH.h"
#include <stdint.h>
#include <limits.h>
#include <math.h>

#define unused(x) ((void) x)

using namespace std;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::ivec2;
using glm::clamp;


#define SCREEN_WIDTH 1000
#define SCREEN_HEIGHT 900
#define FULLSCREEN_MODE false
#define pi 3.1415
//Used to describe a pixel from the image
struct Pixel
{
  int x;
  int y;
  int lx;
  int ly;
  int lzinv;
  float zinv;
  vec4 pos3d;
  float isLit;
};

struct Vertex
{
  vec4 position;
};

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */
void ComputeBoundingBox ( const vector<Pixel>& vertexPixels, int& min_x, int& max_x,
                           int& min_y, int& max_y, int& min_lx, int& max_lx, int& min_ly, int& max_ly);
void BarycentricCoord(Pixel p0, Pixel p1, Pixel p2, Pixel p, vec3& lambda);
void DrawPolygon(const vector<Vertex>& vertices, vec3 currentColor, screen* screen);
void ScreenShader(screen* screen);
void update_rotation_x (float pitch);
void update_rotation_y (float yaw  );
void InterpolatePixels (Pixel a, Pixel b, vector<Pixel>& result);
void VertexShader (const Vertex& v, Pixel& p);
void PixelShader(Pixel& p, screen* screen);
void Update();
void Draw (screen* screen);
void TransformationMatrix ();
glm::vec3 lookUpBuffer (float x, float y);
inline float toLuma(vec3 color);
void FXAA (int x, int y);
void ComputeShadowMap( const vector<Vertex>& vertices, vec3 currentColor, screen* screen );


//Global variables
const vec2 inverseScreenSize (1.0/SCREEN_WIDTH, 1.0/SCREEN_HEIGHT);
vec4 cam_pos(0.0, 0.0, -2.501, 1.0);
vec4 light_pos(0.3, -0.3, -2.2, 1.0);
vec3 lightPower = 34.f*vec3( 1, 1, 1 );
vec3 indirectLightPowerPerArea = 0.5f*vec3( 1, 1, 1 );
vec4 currentNormal(0.0f);
vec3 currentReflectance(0.0f);
float focal_length = SCREEN_HEIGHT/2.0;
float light_focal = SCREEN_HEIGHT/2.0;
std::vector<Triangle> triangles;
float rotation_angle_y = 0.0;
float rotation_angle_x = 0.0;
glm::mat4 R_y = glm::mat4(1.0);
glm::mat4 R_x = glm::mat4(1.0);
glm::mat4 transformation_matrix = glm::mat4(1.0);
float depthBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];
glm::vec3 AA_colorBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];
glm::vec3 colorBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];
Pixel shadowMap[SCREEN_HEIGHT][SCREEN_WIDTH];



int main( int argc, char* argv[] )
{

  screen *screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN_MODE );

  while( NoQuitMessageSDL() )
    {
      Update();
      Draw(screen);
      SDL_Renderframe(screen);
    }

  SDL_SaveImage( screen, "screenshot.bmp" );

  KillSDL(screen);
  return 0;
}

/*Place your drawing here*/
void Draw(screen* screen)
{
  /* Clear buffers */
  memset(screen->buffer, 0, screen->height*screen->width*sizeof(uint32_t));
  memset(colorBuffer, 0, SCREEN_HEIGHT*SCREEN_WIDTH*sizeof(float)*3);
  memset(AA_colorBuffer, 0, SCREEN_HEIGHT*SCREEN_WIDTH*sizeof(float)*3);
  memset(shadowMap, 0, SCREEN_HEIGHT*SCREEN_WIDTH*sizeof(Pixel));


   for( int y=0; y<SCREEN_HEIGHT; ++y )
     for( int x=0; x<SCREEN_WIDTH; ++x )
     {
       depthBuffer[y][x] = 0;

     }
  LoadTestModel(triangles);
  for (uint32_t i=0; i<triangles.size(); i++)
  {
    std::vector<Vertex> vertices(3);

    vertices[0].position = triangles[i].v0;
    vertices[1].position = triangles[i].v1;
    vertices[2].position = triangles[i].v2;

    currentNormal = triangles[i].normal;
    currentReflectance = triangles[i].color;

    //Calculate the projected positions of the triangle vertices
    DrawPolygon( vertices, triangles[i].color, screen );
  }
  ScreenShader(screen);
}


void BarycentricCoord(Pixel p0, Pixel p1, Pixel p2, Pixel p, vec3& lambda) {
  glm::vec2 v0((float)(p1.x - p0.x), (float)(p1.y - p0.y));
  glm::vec2 v1((float)(p2.x - p0.x), (float)(p2.y - p0.y));
  glm::vec2 v2((float)(p.x - p0.x), (float)(p.y - p0.y));
  float d00 = glm::dot(v0, v0);
  float d01 = glm::dot(v0, v1);
  float d11 = glm::dot(v1, v1);
  float d20 = glm::dot(v2, v0);
  float d21 = glm::dot(v2, v1);
  float denom = d00 * d11 - d01 * d01;
  float w2 = (d11 * d20 - d01 * d21) / denom;
  float w3 = (d00 * d21 - d01 * d20) / denom;
  float w1 = 1.0f - w2 - w3;
  if (w1 >= 0 && w2 >= 0 && w3 >= 0 && w1 <= 1 && w2 <= 1 && w3 <= 1){
    lambda = vec3(w1, w2, w3);
  }
  else {
    lambda = vec3(-1, -1, -1);
  }
}

//Calculate the perceived luminosity of a pixel color
inline float toLuma(vec3 color) {
  vec3 luma(0.299, 0.587, 0.114);
  return dot(color, luma);
}

//Map the inverse coordinates onto the color buffer space
glm::vec3 lookUpBuffer (float x, float y) {
  return colorBuffer[clamp((int)(clamp(y, 0.f, 1.f)*SCREEN_HEIGHT-1), 0, SCREEN_HEIGHT-1)]
                    [clamp((int)(clamp(x, 0.f, 1.f)*SCREEN_WIDTH-1), 0, SCREEN_WIDTH-1)];
}

//Parameters to control the FXAA
float EDGE_MIN_THRESHOLD = 1.0f/512.0f;
float EDGE_MAX_THRESHOLD = 1.0f/32.0f;
float GR_SCALE = 2.5f;
float pixel_offset = 1.f/6.f;

//Perform Fast Approximate Anti-Aliasing at a position (x,y) on the rendered image.
void FXAA (int x, int y) {
  vec2 inverseCoord((float)x/(float)SCREEN_WIDTH, (float)y/(float)SCREEN_HEIGHT);

  //Find the mapped coordinates of the corner neighbours in the color buffer.
  vec3 nei_sw = lookUpBuffer(inverseCoord.x - inverseScreenSize.x, inverseCoord.y + inverseScreenSize.y); //Coord of south-west neighbour
  vec3 nei_se = lookUpBuffer(inverseCoord.x + inverseScreenSize.x, inverseCoord.y + inverseScreenSize.y); //Coord of south-east neighbour
  vec3 nei_ne = lookUpBuffer(inverseCoord.x + inverseScreenSize.x, inverseCoord.y - inverseScreenSize.y); //Coord of north-east neighbour
  vec3 nei_nw = lookUpBuffer(inverseCoord.x - inverseScreenSize.x, inverseCoord.y - inverseScreenSize.y); //Coord of north-west neighbour

  //Calculate luma values of the pixel and its diagonal neighbours
  float luma_center = toLuma(lookUpBuffer(inverseCoord.x, inverseCoord.y));
  float luma_sw = toLuma(nei_sw);
  float luma_se = toLuma(nei_se);
  float luma_ne = toLuma(nei_ne);
  float luma_nw = toLuma(nei_nw);

  //Compute the local contrast value. Contrast is strong along the edges of an object.
  float luma_min = glm::min(luma_center, glm::min(glm::min(luma_sw, luma_se), glm::min(luma_nw, luma_ne)));
  float luma_max = glm::max(luma_center, glm::max(glm::max(luma_sw, luma_se), glm::max(luma_nw, luma_ne)));
  float contrast = luma_max - luma_min;

  //Combine the luminosity of the 4 edges around the center
  float luma_left_edge = luma_nw + luma_sw;
  float luma_right_edge = luma_ne + luma_se;
  float luma_south_edge = luma_sw + luma_se;
  float luma_north_edge = luma_nw + luma_ne;

  //Calculate and normalize the luminosity gradient at the current position.
  vec2 grad(-(luma_north_edge - luma_south_edge), (luma_left_edge - luma_right_edge));
  float gr_thresh_val =  glm::max((luma_south_edge + luma_north_edge) * EDGE_MAX_THRESHOLD , EDGE_MIN_THRESHOLD);
  float gr_thresh = 1.0f / (glm::min(abs(grad.x), abs(grad.y)) + gr_thresh_val);
  grad = glm::vec2(glm::min(GR_SCALE, glm::max(-GR_SCALE, grad.x * gr_thresh))*inverseScreenSize.x,
                    glm::min(GR_SCALE, glm::max(-GR_SCALE, grad.y * gr_thresh))*inverseScreenSize.y);

  //Look slightly in the direction of the gradient
  vec2 pos_offset = inverseCoord + (-pixel_offset * grad);
  vec2 neg_offset = inverseCoord + pixel_offset * grad;
  vec2 intermediate1 = inverseCoord + (-0.5f*grad);
  vec2 intermediate2 = inverseCoord + 0.5f*grad;

  //Calculate the average color along the gradient direction
  glm::vec3 gradient_average = 0.5f * (lookUpBuffer(pos_offset.x, pos_offset.y) + lookUpBuffer(neg_offset.x, neg_offset.y));
  glm::vec3 grad_lf = 0.5f * gradient_average + 0.25f * (lookUpBuffer(intermediate1.x, intermediate1.y) + lookUpBuffer(intermediate2.x, intermediate2.y));

  //If the area is too dark, or if the contrast is not big enough, there is no need for anti-aliasing.
  float perceived_luminosity = toLuma(grad_lf) - luma_min;
  if (perceived_luminosity < 0 || perceived_luminosity > contrast)
    AA_colorBuffer[y][x] = gradient_average;
  else AA_colorBuffer[y][x] = grad_lf;
}


void ComputeShadowMap( const vector<Vertex>& vertices, vec3 currentColor, screen* screen ) {
  int V = vertices.size();
  vector<Pixel> lightPixels( V );
  int min_x, min_y, max_x, max_y, min_lx, min_ly, max_lx, max_ly;
  min_x = min_y = max_x = max_y = min_lx = min_ly = max_lx = max_ly = 0;

  //ComputeBoundingBox( lightPixels, min_x, max_x, min_y, max_y, min_lx, max_lx, min_ly, max_ly);

  for( int y=min_y; y<max_y; y++ )
    {
      if (y >= SCREEN_HEIGHT || y < 0 ) continue;
      for( int x=min_x; x<max_x; x++ )
      {
        if(x >= SCREEN_WIDTH || x < 0 ) continue;
        vec3 barycentric_coords;
        Pixel p;
        p.x = x;
        p.y = y;
        BarycentricCoord(lightPixels[0], lightPixels[1], lightPixels[2], p, barycentric_coords);
        if (barycentric_coords.x >= 0 && barycentric_coords.y >= 0 && barycentric_coords.z >= 0
          && barycentric_coords.x <= 1 && barycentric_coords.y <= 1 && barycentric_coords.z <= 1 )
          {
             p.zinv = lightPixels[0].zinv*barycentric_coords.x + lightPixels[1].zinv*barycentric_coords.y
               + lightPixels[2].zinv*barycentric_coords.z;
             p.pos3d = (lightPixels[0].pos3d*barycentric_coords.x*lightPixels[0].zinv
               + lightPixels[1].pos3d*barycentric_coords.y*lightPixels[1].zinv
               + lightPixels[2].pos3d*barycentric_coords.z*lightPixels[2].zinv) / p.zinv;
          }
        }
    }

}

//Draw a 3D polygon
void DrawPolygon( const vector<Vertex>& vertices, vec3 currentColor, screen* screen )
{
  int V = vertices.size();
  vector<Pixel> vertexPixels( V );
  for( int i=0; i<V; ++i ){
    VertexShader( vertices[i], vertexPixels[i] );
  }
  int min_x, min_y, max_x, max_y, min_lx, min_ly, max_lx, max_ly;
  min_x = min_y = max_x = max_y = min_lx = min_ly = max_lx = max_ly = 0;

  ComputeBoundingBox( vertexPixels, min_x, max_x, min_y, max_y, min_lx, max_lx, min_ly, max_ly );

  for( int y=min_y; y<max_y; y++ )
    {
      if (y >= SCREEN_HEIGHT || y < 0 ) continue;
      for( int x=min_x; x<max_x; x++ )
      {
        if(x >= SCREEN_WIDTH || x < 0 ) continue;
        vec3 barycentric_coords;
        Pixel p;
        p.x = x;
        p.y = y;
        BarycentricCoord(vertexPixels[0], vertexPixels[1], vertexPixels[2], p, barycentric_coords);
        if (barycentric_coords.x >= 0 && barycentric_coords.y >= 0 && barycentric_coords.z >= 0
          && barycentric_coords.x <= 1 && barycentric_coords.y <= 1 && barycentric_coords.z <= 1 )
          {
             p.zinv = vertexPixels[0].zinv*barycentric_coords.x + vertexPixels[1].zinv*barycentric_coords.y
               + vertexPixels[2].zinv*barycentric_coords.z;
             p.pos3d = (vertexPixels[0].pos3d*barycentric_coords.x*vertexPixels[0].zinv
               + vertexPixels[1].pos3d*barycentric_coords.y*vertexPixels[1].zinv
               + vertexPixels[2].pos3d*barycentric_coords.z*vertexPixels[2].zinv) / p.zinv;

             PixelShader (p, screen);
          }
        }
    }


}

void ComputeBoundingBox ( const vector<Pixel>& vertexPixels, int& min_x, int& max_x,
                           int& min_y, int& max_y, int& min_lx, int& max_lx, int& min_ly, int& max_ly)
{
  int V = vertexPixels.size();
  int tmp_min_y = numeric_limits<int>::max();
  int tmp_min_x = numeric_limits<int>::max();
  int tmp_max_y = numeric_limits<int>::min();
  int tmp_max_x = numeric_limits<int>::min();

  int tmp_min_ly = numeric_limits<int>::max();
  int tmp_min_lx = numeric_limits<int>::max();
  int tmp_max_ly = numeric_limits<int>::min();
  int tmp_max_lx = numeric_limits<int>::min();


  for ( int i=0; i<V; i++ )
  {
    if (vertexPixels[i].x > tmp_max_x)
      tmp_max_x = vertexPixels[i].x;
    if (vertexPixels[i].x < tmp_min_x)
      tmp_min_x = vertexPixels[i].x;
    if (vertexPixels[i].y > tmp_max_y)
      tmp_max_y = vertexPixels[i].y;
    if (vertexPixels[i].y < tmp_min_y)
      tmp_min_y = vertexPixels[i].y;

    if (vertexPixels[i].lx > tmp_max_lx)
      tmp_max_lx = vertexPixels[i].lx;
    if (vertexPixels[i].lx < tmp_min_lx)
      tmp_min_lx = vertexPixels[i].lx;
    if (vertexPixels[i].ly > tmp_max_ly)
      tmp_max_ly = vertexPixels[i].ly;
    if (vertexPixels[i].ly < tmp_min_ly)
      tmp_min_ly = vertexPixels[i].ly;

  }

  min_x = tmp_min_x;
  min_y = tmp_min_y;
  max_x = tmp_max_x;
  max_y = tmp_max_y;
}



//Project 4D points onto the 2D camera image plane
void VertexShader (const Vertex& v, Pixel& p)
{
  glm::vec4 light_coord = v.position - light_pos;
  glm::vec4 cam_coord = v.position - cam_pos;
  cam_coord = R_y * R_x * cam_coord;
  p.zinv = 1.0f/cam_coord.z;
  p.lzinv = 1.0f/light_coord.z;
  p.x = (int) (focal_length*cam_coord.x/cam_coord.z + SCREEN_WIDTH/2.0);
  p.y = (int) (focal_length*cam_coord.y/cam_coord.z + SCREEN_HEIGHT/2.0);
  p.lx = (int) (light_focal*light_coord.x/light_coord.z + SCREEN_WIDTH/2.0);
  p.ly = (int) (light_focal*light_coord.y/light_coord.z + SCREEN_HEIGHT/2.0);
  p.pos3d = v.position;

}


void PixelShader(Pixel& p, screen* screen)
{
  int x = p.x;
  int y = p.y;
  if (p.zinv >= depthBuffer[y][x])
  {

    //Illumination for each Vertex
    vec4 r = glm::normalize(light_pos - p.pos3d);
    float radius = glm::length(light_pos - p.pos3d);
    vec4 n = glm::normalize(currentNormal);

    float res = glm::dot(r, n);
    float dot = glm::max( res, 0.f );
    float frac = dot / (4.f * pi * radius * radius );

    vec3 D = indirectLightPowerPerArea + frac*lightPower ;
    D = D*currentReflectance ;

     depthBuffer[y][x] = p.zinv;
     colorBuffer[y][x] = D;
   }
}

void ScreenShader(screen* screen)
{
  for (int y=0; y<SCREEN_HEIGHT; y++)
    for(int x=0; x<SCREEN_WIDTH; x++)
      FXAA(x,y);
  for (int y=0; y<SCREEN_HEIGHT; y++)
    for(int x=0; x<SCREEN_WIDTH; x++)
    {
      PutPixelSDL( screen, x, y, AA_colorBuffer[y][x]);
    }
}


//Generate equally-distributed values between two Pixels a and b
void InterpolatePixels (Pixel a, Pixel b, vector<Pixel>& result) {
  int N = result.size();
  float current_x = a.x;
  float current_y = a.y;
  float current_z = a.zinv;
  vec4 current_pos3d(a.pos3d*a.zinv);
  float step_x = (b.x - a.x) / float(glm::max(N-1,1));
  float step_y = (b.y - a.y) / float(glm::max(N-1,1));
  float step_z = (b.zinv - a.zinv) / float(glm::max(N-1,1));
  vec4 pos3d_step = vec4(b.pos3d*b.zinv - a.pos3d*a.zinv) / float(glm::max(N-1,1));
  for (int i=0; i<N; i++)
  {
    {
      result[i].x = current_x;
      result[i].y = current_y;
      result[i].zinv = current_z;
      result[i].pos3d = current_pos3d/current_z;
      current_x += step_x;
      current_y += step_y;
      current_z += step_z;
      current_pos3d += pos3d_step;
    }
  }
}

//Update parameters and calculate rendering time after each frame.
void Update() {
  static int t = SDL_GetTicks();
  /* Compute frame time */
  int t2 = SDL_GetTicks();
  float dt = float(t2-t);
  unused(dt);
  t = t2;

  const uint8_t* keystate = SDL_GetKeyboardState( NULL );
  if( keystate[SDL_SCANCODE_UP] )
  {
    // Move camera forward
    cam_pos.z += 0.05;
  }
  if( keystate[SDL_SCANCODE_DOWN] )
  {
    // Move camera backward
    cam_pos.z -= 0.05;
  }
  if( keystate[SDL_SCANCODE_LEFT] )
  {
    // Move camera to the left
    cam_pos.x -= 0.05;
  }
  if( keystate[SDL_SCANCODE_RIGHT] )
  {
    // Move camera to the right
    cam_pos.x += 0.05;
  }
  if( keystate[SDL_SCANCODE_A] )
  {
    // Rotate camera to the left
    rotation_angle_y -= 0.02;
    update_rotation_y (rotation_angle_y);
  }
  if( keystate[SDL_SCANCODE_D] )
  {
    // Rotate camera to the right
    rotation_angle_y += 0.02;
    update_rotation_y (rotation_angle_y);
  }
  if( keystate[SDL_SCANCODE_W] )
  {
    // Rotate camera uowards
    rotation_angle_x += 0.02;
    update_rotation_x (rotation_angle_x);
  }
  if( keystate[SDL_SCANCODE_S] )
  {
    // Rotate camera downwards
    rotation_angle_x -= 0.02;
    update_rotation_x (rotation_angle_x);
  }

  //std::cout << " time: " << dt << " ms." << std::endl;
}

//Rotate the camera view around the Y axis.
void update_rotation_y (float yaw) {
  R_y =  glm::mat4 (cos(yaw), 0, sin(yaw), 0,
                       0,     1,     0,    0,
                   -sin(yaw), 0, cos(yaw), 0,
                       0,     0,     0,    1);
}

//Rotate the camera view around the X axis.
void update_rotation_x (float pitch) {
  R_x =  glm::mat4 (1,     0,          0,       0,
                    0, cos(pitch), -sin(pitch), 0,
                    0, sin(pitch),  cos(pitch), 0,
                    0,     0,          0,       1);
}

//Create a homogeneous-coordinates transformation matrix for translation and rotation

void TransformationMatrix() {
  glm::mat4 cam_transform = glm::mat4 (1, 0, 0, cam_pos.x,
                                       0, 1, 0, cam_pos.y,
                                       0, 0, 1, cam_pos.z,
                                       0, 0, 0,     1     );

  glm::mat4 cam_transform_r = glm::mat4 (1, 0, 0, -cam_pos.x,
                                         0, 1, 0, -cam_pos.y,
                                         0, 0, 1, -cam_pos.z,
                                         0, 0, 0,     1     );

  transformation_matrix = cam_transform * R_y * R_x * cam_transform_r;
}
