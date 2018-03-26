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


#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 256
#define FULLSCREEN_MODE false

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */

void Interpolate (ivec2 a, ivec2 b, vector<ivec2>& result);
void VertexShader (const vec4& v, ivec2& p);
void Update();
void Draw(screen* screen);
//void TransformationMatrix ();

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

    vertices[0] = triangles[i].v0;
    vertices[1] = triangles[i].v1;
    vertices[2] = triangles[i].v2;

    for (int v=0; v<3; v++)
    {
      ivec2 projPos;
      VertexShader( vertices[v], projPos);
      vec3 colour(1.0,1.0,1.0);
      PutPixelSDL( screen, projPos.x, projPos.y, colour );
    }

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

void VertexShader (const vec4& v, ivec2& p)
{
  float frac = focal_length/v.z;
  float x = frac*v.x + SCREEN_WIDTH/2.0;
  float y = frac*v.y + SCREEN_HEIGHT/2.0;
  p.x = round(x);
  p.y = round(y);
}

void Interpolate (ivec2 a, ivec2 b, vector<ivec2>& result)
{
  int N = result.size();
  vec2 step = vec2(b-a) / float(max(N-1,1));
  vec2 current(a);
  for (int i=0; i<N; i++)
  {
    result[i] = current;
    current += step;
  }
}
