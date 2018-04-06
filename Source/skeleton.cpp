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

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */
void ComputePolygonRows (const vector<glm::ivec2>& vertexPixels, vector<glm::ivec2>& leftPixels, vector<glm::ivec2>& rightPixels);
void DrawRows (const vector<ivec2>& leftPixels, const vector<ivec2>& rightPixels, vec3 currentColor, screen* screen);
void DrawPolygon( const vector<vec4>& vertices, vec3 currentColor, screen* screen);
void update_rotation_x (float pitch);
void update_rotation_y (float yaw  );
void Interpolate (glm::ivec2 a, glm::ivec2 b, vector<glm::ivec2>& result);
void ComputePolygonRows (const vector<glm::ivec2>& vertexPixels, vector<glm::ivec2>& leftPixels, vector<glm::ivec2>& rightPixels );
void VertexShader (const glm::vec4& v, glm::ivec2& p);
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
  /* Clear buffer */
  memset(screen->buffer, 0, screen->height*screen->width*sizeof(uint32_t));
  LoadTestModel(triangles);
  //printf("#%d triangles\n", triangles.size());
  for (int i=0; i<triangles.size(); i++)
  {
    std::vector<vec4> vertices(3);
    std::vector<glm::ivec2> projections(3);

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
  vector<ivec2> vertexPixels( V );
  for( int i=0; i<V; ++i )
    VertexShader( vertices[i], vertexPixels[i] );
  vector<ivec2> leftPixels;
  vector<ivec2> rightPixels;
  ComputePolygonRows( vertexPixels, leftPixels, rightPixels );
  DrawRows( leftPixels, rightPixels, currentColor, screen );
}

void ComputePolygonRows (const vector<glm::ivec2>& vertexPixels, vector<glm::ivec2>& leftPixels, vector<glm::ivec2>& rightPixels)
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
    glm::ivec2 leftPixel  = glm::ivec2(numeric_limits<int>::max(), (min_y + i));
    glm::ivec2 rightPixel = glm::ivec2(numeric_limits<int>::min(), (min_y + i));
    leftPixels.push_back(leftPixel);
    rightPixels.push_back(rightPixel);
  }

  for ( int i=0; i<V; i++ )
  {
    int j = (i + 1)%V;
    glm::ivec2 delta = glm::abs( vertexPixels[i] - vertexPixels[j] );
    int pixels  = glm::max( delta.x, delta.y ) + 1;
    vector<glm::ivec2> edge( pixels );
    Interpolate( vertexPixels[i], vertexPixels[j], edge );

    for (int px = 0; px<pixels; px++)
    {
      int y_idx = edge[px].y - min_y + 1;
      if(edge[px].x < leftPixels[y_idx].x)
        leftPixels[y_idx].x = edge[px].x;
      if(edge[px].x > rightPixels[y_idx].x)
        rightPixels[y_idx].x = edge[px].x;
    }
  }
}

void DrawRows (const vector<ivec2>& leftPixels, const vector<ivec2>& rightPixels, vec3 currentColor, screen* screen)
{
  //printf("%s\n", "DrawRows started");
  for (int i = 0; i<leftPixels.size(); i++)
  {
    int pixels = rightPixels[i].x - leftPixels[i].x + 1;
    vector<glm::ivec2> line( pixels );
    Interpolate( leftPixels[i], rightPixels[i], line );
    for (int pixel = 0; pixel<pixels; pixel++)
    {
      PutPixelSDL( screen, line[pixel].x, line[pixel].y, currentColor);
    }
  }
    //printf("%s\n", "DrawRows ended");
}

//Project 4D points onto the 2D camera image plane
void VertexShader (const vec4& v, glm::ivec2& p)
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

  p.x = round(x);
  p.y = round(y);
}

//Generate equally-distributed values between two points a and b
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
