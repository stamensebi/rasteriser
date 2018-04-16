#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModelH.h"
#include <stdint.h>
#include <limits.h>
#include <math.h>

using namespace std;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::ivec2;


#define SCREEN_WIDTH 960
#define SCREEN_HEIGHT 768
#define FULLSCREEN_MODE false
#define pi 3.1415
//Used to describe a pixel from the image
struct Pixel
{
  int x;
  int y;
  float zinv;
  vec3 illumination;
};

struct Vertex
{
  vec4 position;
  vec4 normal;
  vec3 reflectance;
};

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */
void ComputePolygonRows (const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels);
void DrawRows (const vector<Pixel>& leftPixels, const vector<Pixel>& rightPixels, vec3 currentColor, screen* screen);
void DrawPolygon( const vector<Vertex>& vertices, vec3 currentColor, screen* screen);
void update_rotation_x (float pitch);
void update_rotation_y (float yaw  );
void InterpolatePixels (Pixel a, Pixel b, vector<Pixel>& result);
void Interpolate (glm::ivec2 a, glm::ivec2 b, vector<glm::ivec2>& result);
void VertexShader (const Vertex& v, Pixel& p);
void PixelShader( const Pixel& p, screen* screen);
void Update();
void Draw (screen* screen);
void TransformationMatrix (glm::mat4 tr_mat, glm::vec4 camera_position, glm::mat4 rotation_matrix);

//Global variables
vec4 cam_pos(0.0, 0.0, -2.501, 1.0);
vec4 light_pos(0, -0.5, -0.7, 1.0);
vec3 lightPower = 34.f*vec3( 1, 1, 1 );
vec3 indirectLightPowerPerArea = 0.5f*vec3( 1, 1, 1 );
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
  //memset(depthBuffer, 0, SCREEN_HEIGHT*SCREEN_WIDTH*sizeof(float));
   for( int y=0; y<SCREEN_HEIGHT; ++y )
     for( int x=0; x<SCREEN_WIDTH; ++x )
     depthBuffer[y][x] = 0;

  LoadTestModel(triangles);
  for (uint32_t i=0; i<triangles.size(); i++)
  {
    std::vector<Vertex> vertices(3);

    vertices[0].position = triangles[i].v0;
    vertices[1].position = triangles[i].v1;
    vertices[2].position = triangles[i].v2;
    vertices[0].normal = triangles[i].normal;
    vertices[1].normal = triangles[i].normal;
    vertices[2].normal = triangles[i].normal;
    vertices[0].reflectance = triangles[i].color;
    vertices[1].reflectance = triangles[i].color;
    vertices[2].reflectance = triangles[i].color;


    //Calculate the projected positions of the triangle vertices
    DrawPolygon( vertices, triangles[i].color, screen );
    //printf("triangle #%d\n", i);
  }

}

//Draw a 3D polygon
void DrawPolygon( const vector<Vertex>& vertices, vec3 currentColor, screen* screen )
{
  int V = vertices.size();
  vector<Pixel> vertexPixels( V );
  for( int i=0; i<V; ++i )
    VertexShader( vertices[i], vertexPixels[i] );
  vector<Pixel> leftPixels;
  vector<Pixel> rightPixels;
  //cout << "size before: " << leftPixels.size()<<endl;
  ComputePolygonRows( vertexPixels, leftPixels, rightPixels );
  //cout << "color= " << rightPixels[0].illumination.x << " " << rightPixels[0].illumination.y << " " << rightPixels[0].illumination.z << " " << endl;
  //cout << "size AFTER 1: " << leftPixels.size()<<endl;
  DrawRows( leftPixels, rightPixels, currentColor, screen );
  //cout << "size AFTER 2: " << leftPixels.size()<<endl;
}


void ComputePolygonRows ( const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels )
{
  int V = vertexPixels.size();

  // Find max and min y-value of the polygon
  // and compute the number of rows it occupies.
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
  //cout<<"MIN Y IS!!!"<< min_y<<endl;
  //Resize vectors to ROWS
  leftPixels.resize(ROWS);
  rightPixels.resize(ROWS);

  // Initialize the x-coordinates in leftPixels
  // to some really large value and the x-coordinates
  // in rightPixels to some really small value.
  for (int i=0; i<ROWS; i++)
  {
    /*Pixel leftPixel;//  = new Pixel();
    Pixel rightPixel;// = new Pixel();
    leftPixels.x = numeric_limits<int>::max();
    leftPixel.y = min_y + i;
    leftPixel.zinv = 0;
    rightPixel.x = numeric_limits<int>::min();
    rightPixel.y = min_y + i;
    rightPixel.zinv = 0;*/
    leftPixels[i].x = numeric_limits<int>::max();
    leftPixels[i].y = min_y + i;
    leftPixels[i].zinv = 0;
    //leftPixels[i].illumination = vertexPixels[0].illumination;

    rightPixels[i].x = numeric_limits<int>::min();
    rightPixels[i].y = min_y + i;
    rightPixels[i].zinv = 0;
    //rightPixels[i].illumination = vertexPixels[0].illumination;


    //leftPixels.push_back(leftPixel);
    //rightPixels.push_back(rightPixel);
  }
  // Loop through all edges of the polygon and use
  // linear interpolation to find the x-coordinate for
  // each row it occupies. Update the corresponding
  // values in rightPixels and leftPixels.
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
      int y_idx = edge[px].y - min_y ;
      if(y_idx<0) y_idx = 0;
      //cout << "Y IDX: " << y_idx<<endl;
      if(edge[px].x < leftPixels[y_idx].x)
      {
        leftPixels[y_idx].x = edge[px].x;
        //leftPixels[y_idx].y =
        leftPixels[y_idx].zinv = edge[px].zinv;
        leftPixels[y_idx].illumination = edge[px].illumination;
      }
      if(edge[px].x > rightPixels[y_idx].x)
      {
        rightPixels[y_idx].x = edge[px].x;
        rightPixels[y_idx].zinv = edge[px].zinv;
        rightPixels[y_idx].illumination = edge[px].illumination;
      }
    }
  }

}

void DrawRows (const vector<Pixel>& leftPixels, const vector<Pixel>& rightPixels, vec3 currentColor, screen* screen)
{
  int P = leftPixels.size();
  for (uint32_t i = 0; i<P; i++)
  {
    int pixels = rightPixels[i].x - leftPixels[i].x + 1;
    vector<Pixel> line( pixels );
    //cout << "PIXELS " << line.size()<<endl;
    InterpolatePixels( leftPixels[i], rightPixels[i], line );
    //cout<<"INTERPOLATE WORKED "<< line.size()<<endl;
    for (int pixel = 0; pixel<pixels; pixel++)
    {
      //PutPixelSDL( screen, line[pixel].x, line[pixel].y, vec3(0,0,0));
      //PixelShader( line[pixel], screen );
      //if(line[pixel].y > 0 && line[pixel.x] > 0)
      if(line[pixel].zinv > depthBuffer[line[pixel].y][line[pixel].x])
        {
          depthBuffer[line[pixel].y][line[pixel].x] = line[pixel].zinv;
          //cout << "p.illumination " << line[pixel].illumination.x << " " << line[pixel].illumination.y << " " << line[pixel].illumination.z << endl;
          // if(line[pixel].illumination.x > 0 && line[pixel].illumination.y > 0 && line[pixel].illumination.z > 0) {
          //   PutPixelSDL( screen, line[pixel].x, line[pixel].y, currentColor * line[pixel].illumination);
          // }
          // else {
          //   PutPixelSDL( screen, line[pixel].x, line[pixel].y, currentColor );
          // }
          PutPixelSDL( screen, line[pixel].x, line[pixel].y,  line[pixel].illumination);
        }
    }
  }
}

//Project 4D points onto the 2D camera image plane
void VertexShader (const Vertex& v, Pixel& p)
{
  glm::vec4 cam_coord = v.position - cam_pos;
  cam_coord = R_y * R_x * cam_coord;

  p.zinv = 1.0/cam_coord.z;
  p.x = round(focal_length*p.zinv*cam_coord.x + SCREEN_WIDTH/2.0);
  p.y = round(focal_length*p.zinv*cam_coord.y + SCREEN_HEIGHT/2.0);

  //Illumination for each Vertex
  vec4 r = glm::normalize(light_pos - v.position);
  float radius = length(light_pos - v.position);
  vec4 n = glm::normalize(v.normal);
  /*n.x *= 0.8;
  n.y *= 0.8;
  n.z *= 0.8;*/
  float res = glm::dot(r, n);
  float dot = glm::max( res, 0.f );
  float frac = dot / (4.f * pi * radius * radius );
  cout << " frac " << frac << endl;
  vec3 D = indirectLightPowerPerArea + frac*lightPower ;
  p.illumination =  v.reflectance * D;

}


void PixelShader( const Pixel& p, screen* screen)
{
  int x = p.x;
  int y = p.y;
  //cout << x << " " << y << endl;
   if (p.zinv >= depthBuffer[y][x])
   {
     depthBuffer[y][x] = p.zinv;
     PutPixelSDL( screen, x, y, p.illumination);
   }
}

//Generate equally-distributed values between two Pixels a and b
void InterpolatePixels (Pixel a, Pixel b, vector<Pixel>& result)
{
  int N = result.size();
  float step_x = (b.x - a.x) / float(glm::max(N-1,1));
  float step_y = (b.y - a.y) / float(glm::max(N-1,1));
  float step_z = (b.zinv - a.zinv) / float(glm::max(N-1,1));
  vec3 illumination_step = vec3(b.illumination - a.illumination) / float(glm::max(N-1,1));
  float current_x = a.x;
  float current_y = a.y;
  float current_z = a.zinv;
  vec3 current_ill(a.illumination);
  for (int i=0; i<N; i++)
  {
    result[i].x = current_x;
    result[i].y = current_y;
    result[i].zinv = current_z;
    result[i].illumination = current_ill;
    current_x += step_x;
    current_y += step_y;
    current_z += step_z;
    current_ill += illumination_step;
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

  //std::cout << "Render time: " << dt << " ms." << std::endl;
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
