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


#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 256
#define FULLSCREEN_MODE false

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */

void Interpolate (glm::ivec2 a, glm::ivec2 b, vector<glm::ivec2>& result);
void VertexShader (const glm::vec4& v, glm::ivec2& p);
void DrawLineSDL (SDL_Surface* surface, glm::ivec2 a, glm::ivec2 b, vec3 color);
void Update();
void Draw(screen* screen);
void TransformationMatrix ( glm::mat4 tr_mat, glm::vec4 camera_position, glm::mat3 rotation_matrix);
void DrawPolygonEdges ( screen* screen, const vector<glm::vec4>& vertices );

//Global variables
vec4 cam_pos(0.0, 0.0, -3.001, 1.0);
float focal_length = SCREEN_HEIGHT/2.0;
std::vector<Triangle> triangles;

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

  for (uint32_t i=0; i<triangles.size(); i++)
  {
    std::vector<vec4> vertices(3);
    std::vector<glm::ivec2> projections(3);

    vertices[0] = triangles[i].v0;
    vertices[1] = triangles[i].v1;
    vertices[2] = triangles[i].v2;

    //Calculate the projected positions of the triangle vertices
    DrawPolygonEdges(screen, vertices);

  }

}

//Draw a line between two 4D points
void DrawLineSDL (screen* surface, glm::ivec2 a, glm::ivec2 b, vec3 color)
{
  glm::ivec2 delta = glm::abs( a-b );
  int pixels  = glm::max( delta.x, delta.y ) + 1;
  vector<glm::ivec2> line( pixels );
  Interpolate( a, b, line );
  for (int pixel = 0; pixel<pixels; pixel++)
  {
    PutPixelSDL( surface, line[pixel].x, line[pixel].y, color);
  }
}

void DrawPolygonEdges ( screen* screen, const vector<glm::vec4>& vertices )
{
  int V = vertices.size();

  vector<glm::ivec2> projections( V );
  for ( int i=0; i<V; i++ )
    {
      VertexShader( vertices[i], projections[i] );
    }
  for ( int a_idx=0; a_idx<V; a_idx++ )
  {
    int b_idx = (a_idx + 1)%V;
    glm::vec3 color(1.0, 1.0, 1.0);
    DrawLineSDL( screen, projections[a_idx], projections[b_idx], color );
  }
}

/*Place updates of parameters here*/
void Update()
{
  static int t = SDL_GetTicks();
  /* Compute frame time */
  int t2 = SDL_GetTicks();
  float dt = float(t2-t);
  t = t2;
  /*Good idea to remove this*/
  std::cout << "Render time: " << dt << " ms." << std::endl;
  /* Update variables*/
}

void VertexShader (const vec4& v, glm::ivec2& p)
{
  //Can create a camera movement matrix using TransformationMatrix(), then
  //multiply by v to the right to transform the image position.
  float frac = focal_length/v.z;
  float x = frac*v.x + SCREEN_WIDTH/2.0;
  float y = frac*v.y + SCREEN_HEIGHT/2.0;

  p.x = round(x);
  p.y = round(y);
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

//Create a homogeneous-coordinates transformation matrix for translation and rotation
void TransformationMatrix ( glm::mat4 transformation_mat, glm::vec4 camera_position, glm::mat3 rotation_matrix)
{
  //Create each row of the camera transform matrix. Only done outside for readability
  glm::vec4 cam_x_col = glm::vec4( glm::vec3( 0.0 ), camera_position.x);
  glm::vec4 cam_y_col = glm::vec4( glm::vec3( 0.0 ), camera_position.y);
  glm::vec4 cam_z_col = glm::vec4( glm::vec3( 0.0 ), camera_position.z);
  glm::vec4 cam_t_col = glm::vec4( glm::vec3( 0.0 ), 1.0);

  //Expand the camera position vector into a 4x4 homogeneous transformation
  glm::mat4 cam_transform   = glm::mat4( cam_x_col, cam_y_col, cam_z_col, cam_t_col);
  glm::mat4 cam_transform_r = glm::mat4( -cam_x_col, -cam_y_col, -cam_z_col, cam_t_col);


  vec4 rotation_matrix_x = vec4(rotation_matrix[0][0], rotation_matrix[1][0], rotation_matrix[2][0], 0.0);
  vec4 rotation_matrix_y = vec4(rotation_matrix[0][1], rotation_matrix[1][1], rotation_matrix[1][2], 0.0);
  vec4 rotation_matrix_z = vec4(rotation_matrix[0][2], rotation_matrix[1][2], rotation_matrix[2][2], 0.0);  
  //Expand the rotation matrix to a 4x4 homogeneous transformation
  glm::mat4 homogeneous_rotate = glm::mat4(rotation_matrix_x,
                                           rotation_matrix_y,
                                           rotation_matrix_z,
                                           glm::vec4(cam_t_col));

  transformation_mat = cam_transform * homogeneous_rotate * cam_transform_r;
}
