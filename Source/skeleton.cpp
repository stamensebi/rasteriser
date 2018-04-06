#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModelH.h"
#include <stdint.h>
#include <limits.h>
#include <math.h>

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec4;
using glm::mat4;
using glm::vec2;
using glm::ivec2;


#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 256
#define FULLSCREEN_MODE false

//Used to describe a pixel from the image
struct Pixel
{
  int x;
  int y;
  float zinv;
};

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */
void ComputePolygonRows (const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels);
void DrawRows (const vector<Pixel>& leftPixels, const vector<Pixel>& rightPixels, vec3 currentColor, screen* screen);
void DrawPolygon( const vector<vec4>& vertices, vec3 currentColor, screen* screen);
void update_rotation_x (float pitch);
void update_rotation_y (float yaw  );
void InterpolatePixels (Pixel a, Pixel b, vector<Pixel>& result);
void Interpolate (glm::ivec2 a, glm::ivec2 b, vector<glm::ivec2>& result);
void VertexShader (const glm::vec4& v, Pixel& p);
void Update();
void Draw (screen* screen);
void TransformationMatrix (glm::mat4 tr_mat, glm::vec4 camera_position, glm::mat4 rotation_matrix);

//Global variables
vec4 cam_pos(0.0, 0.0, -3.001, 1.0);
float focal_length = SCREEN_HEIGHT/2.0;
std::vector<Triangle> triangles;
float rotation_angle_y = 0.0;
float rotation_angle_x = 0.0;
glm::mat4 R_y = glm::mat4(1.0);
glm::mat4 R_x = glm::mat4(1.0);
float depthBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];

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
  for( int y=0; y<SCREEN_HEIGHT; ++y )
    for( int x=0; x<SCREEN_WIDTH; ++x )
      depthBuffer[y][x] = 0;

  LoadTestModel(triangles);
  //printf("#%d triangles\n", triangles.size());
  for (uint32_t i=0; i<triangles.size(); i++)
  {
    std::vector<vec4> vertices(3);

    vertices[0] = triangles[i].v0;
    vertices[1] = triangles[i].v1;
    vertices[2] = triangles[i].v2;

    //Calculate the projected positions of the triangle vertices
    DrawPolygon( vertices, triangles[i].color, screen );
    //printf("triangle #%d\n", i);
  }

}

//Draw a 3D polygon
void DrawPolygon( const vector<vec4>& vertices, vec3 currentColor, screen* screen )
{
  int V = vertices.size();
  vector<Pixel> vertexPixels( V );
  for( int i=0; i<V; ++i )
    VertexShader( vertices[i], vertexPixels[i] );
  vector<Pixel> leftPixels;
  vector<Pixel> rightPixels;
  ComputePolygonRows( vertexPixels, leftPixels, rightPixels );
  DrawRows( leftPixels, rightPixels, currentColor, screen );
}


void ComputePolygonRows ( const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels )
{
  int V = vertexPixels.size();
  int min_y = numeric_limits<int>::max();
  int max_y = numeric_limits<int>::min();
  for ( int i=0; i<V; i++ )
  {
    if (vertexPixels[i].y > max_y)
      max_y = vertexPixels[i].y;
    if (vertexPixels[i].y < min_y)
      min_y = vertexPixels[i].y;
  }
  int ROWS = max_y - min_y + 1;

  for (int i=0; i<ROWS; i++)
  {
    Pixel leftPixel;//  = new Pixel();
    Pixel rightPixel;// = new Pixel();
    leftPixel.x = numeric_limits<int>::max();
    leftPixel.y = min_y + i;
    rightPixel.x = numeric_limits<int>::min();
    rightPixel.y = min_y + i;
    leftPixels.push_back(leftPixel);
    rightPixels.push_back(rightPixel);
  }

  for ( int i=0; i<V; i++ )
  {
    int j = (i + 1)%V;
    int delta_x = glm::abs(vertexPixels[i].x - vertexPixels[j].x);
    int delta_y = glm::abs(vertexPixels[i].y - vertexPixels[j].y);
    int pixels  = glm::max( delta_x, delta_y ) + 1;
    vector<Pixel> edge( pixels );
    InterpolatePixels( vertexPixels[i], vertexPixels[j], edge );

    for (int px = 0; px<pixels; px++)
    {
      int y_idx = edge[px].y - min_y + 1;
      if(edge[px].x < leftPixels[y_idx].x)
      {
        leftPixels[y_idx].x = edge[px].x;
        leftPixels[y_idx].zinv = edge[px].zinv;
      }
      if(edge[px].x > rightPixels[y_idx].x)
      {
        rightPixels[y_idx].x = edge[px].x;
        rightPixels[y_idx].zinv = edge[px].zinv;
      }
    }
  }
}

void DrawRows (const vector<Pixel>& leftPixels, const vector<Pixel>& rightPixels, vec3 currentColor, screen* screen)
{
  for (uint32_t i = 0; i<leftPixels.size(); i++)
  {
    int pixels = rightPixels[i].x - leftPixels[i].x + 1;
    vector<Pixel> line( pixels );
    printf("%s\n", "boom1");
    InterpolatePixels( leftPixels[i], rightPixels[i], line );
    printf("%d\n", line[pixels-1].x);
    for (int pixel = 0; pixel<pixels; pixel++)
    {
      if(line[pixel].zinv > depthBuffer[line[pixel].y][line[pixel].x])
        {
          PutPixelSDL( screen, line[pixel].x, line[pixel].y, currentColor);
          depthBuffer[line[pixel].y][line[pixel].x] = line[pixel].zinv;
        }
    }
  }
}

//Project 4D points onto the 2D camera image plane
void VertexShader (const vec4& v, Pixel& p)
{
  glm::vec4 cam_coord = v - cam_pos;
  cam_coord = R_y * R_x * cam_coord;

  //Testing only.
  /**  glm::mat4 tr_mat  = glm::mat4();
  glm::mat4 rot_mat = glm::mat4( 1.0 );
  TransformationMatrix( tr_mat, cam_pos, rot_mat );
  glm::vec4 test = tr_mat * v;**/

  float frac = focal_length/cam_coord.z;
  float x = frac*cam_coord.x + SCREEN_WIDTH/2.0;
  float y = frac*cam_coord.y + SCREEN_HEIGHT/2.0;

  p.zinv = 1/cam_coord.z;
  p.x = round(x);
  p.y = round(y);
}

//Generate equally-distributed values between two Pixels a and b
void InterpolatePixels (Pixel a, Pixel b, vector<Pixel>& result)
{
  int N = result.size();
  int step_x = (b.x - a.x) / float(max(N-1,1));
  int step_y = (b.y - a.y) / float(max(N-1,1));
  int step_z = (b.zinv - a.zinv) / float(max(N-1,1));
  int current_x = a.x;
  int current_y = a.y;
  int current_z = a.zinv;
  for (int i=0; i<N; i++)
  {
    result[i].x = current_x;
    result[i].y = current_y;
    result[i].zinv = current_z;
    current_x += step_x;
    current_y += step_y;
    current_z += step_z;
  }
}

void Interpolate (glm::ivec2 a, glm::ivec2 b, vector<glm::ivec2>& result)
{
  int N = result.size();
  glm::vec2 step = vec2(b-a) / float(max(N-1,1));
  glm::vec2 current(a);
  for (int i=0; i<N; i++)
  {
    result[i] = current;
    current += step;
  }
}

//Update parameters and calculate rendering time after each frame.
void Update()
{
  static int t = SDL_GetTicks();
  /* Compute frame time */
  int t2 = SDL_GetTicks();
  float dt = float(t2-t);
  t = t2;

  const uint8_t* keystate = SDL_GetKeyboardState( NULL );
  if( keystate[SDL_SCANCODE_UP] )
  {
    // Move camera forward
    cam_pos.z += 0.1;
  }
  if( keystate[SDL_SCANCODE_DOWN] )
  {
    // Move camera backward
    cam_pos.z -= 0.1;
  }
  if( keystate[SDL_SCANCODE_LEFT] )
  {
    // Move camera to the left
    cam_pos.x -= 0.1;
  }
  if( keystate[SDL_SCANCODE_RIGHT] )
  {
    // Move camera to the right
    cam_pos.x += 0.1;
  }
  if( keystate[SDL_SCANCODE_A] )
  {
    // Rotate camera to the left
    rotation_angle_y -= 0.05;
    update_rotation_y (rotation_angle_y);
  }
  if( keystate[SDL_SCANCODE_D] )
  {
    // Rotate camera to the right
    rotation_angle_y += 0.05;
    update_rotation_y (rotation_angle_y);
  }
  if( keystate[SDL_SCANCODE_W] )
  {
    // Rotate camera uowards
    rotation_angle_x += 0.05;
    update_rotation_x (rotation_angle_x);
  }
  if( keystate[SDL_SCANCODE_S] )
  {
    // Rotate camera downwards
    rotation_angle_x -= 0.05;
    update_rotation_x (rotation_angle_x);
  }

  std::cout << "Render time: " << dt << " ms." << std::endl;
}

//Rotate the camera view around the Y axis.
void update_rotation_y (float yaw)
{
  R_y =  glm::mat4 (cos(yaw), 0, sin(yaw), 0,
                       0,     1,     0,    0,
                   -sin(yaw), 0, cos(yaw), 0,
                       0,     0,     0,    1);
}

//Rotate the camera view around the X axis.
void update_rotation_x (float pitch)
{
  R_x =  glm::mat4 (1,     0,          0,       0,
                    0, cos(pitch), -sin(pitch), 0,
                    0, sin(pitch),  cos(pitch), 0,
                    0,     0,          0,       1);
}

//Create a homogeneous-coordinates transformation matrix for translation and rotation
void TransformationMatrix ( glm::mat4 transformation_mat, glm::vec4 camera_position, glm::mat4 rotation_matrix)
{
  //Create each row of the camera transform matrix. Only done outside for readability
  glm::vec4 cam_x_col = glm::vec4( glm::vec3( 0.0 ), camera_position.x );
  glm::vec4 cam_y_col = glm::vec4( glm::vec3( 0.0 ), camera_position.y );
  glm::vec4 cam_z_col = glm::vec4( glm::vec3( 0.0 ), camera_position.z );
  glm::vec4 cam_t_col = glm::vec4( glm::vec3( 0.0 ), 1.0 );

  //Expand the camera position vector into a 4x4 homogeneous transformation
  glm::mat4 cam_transform   = glm::mat4(  cam_x_col,  cam_y_col,  cam_z_col, cam_t_col );
  glm::mat4 cam_transform_r = glm::mat4( -cam_x_col, -cam_y_col, -cam_z_col, cam_t_col );


  vec4 rotation_matrix_x = vec4(rotation_matrix[0][0], rotation_matrix[1][0], rotation_matrix[2][0], 0.0);
  vec4 rotation_matrix_y = vec4(rotation_matrix[0][1], rotation_matrix[1][1], rotation_matrix[2][1], 0.0);
  vec4 rotation_matrix_z = vec4(rotation_matrix[0][2], rotation_matrix[1][2], rotation_matrix[2][2], 0.0);
  //Expand the rotation matrix to a 4x4 homogeneous transformation
  glm::mat4 homogeneous_rotate = glm::mat4(rotation_matrix_x,
                                           rotation_matrix_y,
                                           rotation_matrix_z,
                                           cam_t_col);

  transformation_mat = cam_transform * homogeneous_rotate * cam_transform_r;
}
