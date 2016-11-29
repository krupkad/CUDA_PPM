#define _USE_MATH_DEFINES
#include <cstdio>
#include <cmath>
#include <ctime>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <cuda_runtime.h>

#include "dcel.hpp"
#include "shader.hpp"
#include "util/error.hpp"

// Standard glut-based program functions
void resizeCallback(GLFWwindow*, int, int);
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void scrollCallback(GLFWwindow* window, double dx, double dy);
void updateCamera();
int xSize = 1024, ySize = 800;
float lastX, lastY, xpos, ypos;
bool leftMousePressed = false, rightMousePressed = false;
float fovy = (float) (M_PI / 4), zNear = 0.10f, zFar = 100.0f;
float theta = 1.22f, phi = -0.70f, zoom = 5.0f;
glm::vec3 camPos;
glm::mat4 projection;

Shader *shader;
DCEL *dcel;

int cudaProbe() {
	int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    checkCUDAError("cudaGetDeviceProperties", __LINE__);
    
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Compute capability: %d.%d", prop.major, prop.minor);
    if (prop.major < 3) {
      printf(" (< 3.0, disabling texSamp)");
      dcel->canUseTexObjs = false;
    } else {
      dcel->canUseTexObjs = true;
    }
    printf("\n");
  }
	return nDevices;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("usage: %s nBasis nSamp objFile\n", argv[0]);
    return 0;
  }
  int nBasis = atoi(argv[1]);
  int nSamp = atoi(argv[2]);

  // initialize window
  GLFWwindow *window = nullptr;
  if(!glfwInit()) {
    printf("couldn't initialize glfw, disabling visualization\n");
  } else {
    window = glfwCreateWindow(xSize, ySize, "Scene Graph", NULL, NULL);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwMakeContextCurrent(window);
    printf("initialized glfw\n");

    // initialize glew
    glewInit();

    // Set the color which clears the screen between frames
    glClearColor(0, 0, 0, 1);

    // Enable and clear the depth buffer
    glEnable(GL_DEPTH_TEST);
    glClearDepth(1.0);
    glDepthFunc(GL_LEQUAL);

    // set glfw callbacks
    glfwSetWindowSizeCallback(window, resizeCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
  
    //create our shader
    shader = new Shader();
    shader->setShader("shaders/dcel.frag.glsl", GL_FRAGMENT_SHADER);
    shader->setShader("shaders/dcel.vert.glsl", GL_VERTEX_SHADER);
    shader->recompile();

    updateCamera();
    printGLErrorLog();
  }  
  dcel = new DCEL(argv[3], window != nullptr);
  
  // check CUDA devices
  int nDevices = cudaProbe();
  if (!nDevices) {
    printf("no CUDA device found\n");
    return 0;
  }

  // actually build the dcel
  dcel->rebuild(nBasis, nSamp);
  printf("created dcel\n");

  int nbFrames = 0;
  double lastTime = window ? glfwGetTime() : double(clock())/CLOCKS_PER_SEC;
  double dt = 0, alpha = .98;
  while(!(window && glfwWindowShouldClose(window))) {
    // update the DCEL
    dt += dcel->update();
    
    // draw if we are visualizing
    if (window) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      dcel->draw(shader, nullptr);

      // Move the rendering we just made onto the screen
      glfwSwapBuffers(window);
      glfwPollEvents();

      // Check for any GL errors that have happened recently
      printGLErrorLog();
    }

    // perform timing
    double currentTime = window ? glfwGetTime() : double(clock())/CLOCKS_PER_SEC;
    nbFrames++;
    if (currentTime - lastTime >= 1.0){
       printf("%.1f fps (dt = %.3g ms)\n", double(nbFrames)/(currentTime - lastTime), dt/nbFrames);
       nbFrames = 0;
       dt = 0.0f;
       lastTime += 1.0;
    }
  }

  if (window) {
    glfwDestroyWindow(window);
    glfwTerminate();
    delete shader;
  }
  delete dcel;

  return 0;
}

void resizeCallback(GLFWwindow* window, int x, int y) {
  xSize = x;
  ySize = y;
  glViewport(0,0,x,y);
  updateCamera();
}

void errorCallback(int error, const char *description) {
  fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
  if (key == GLFW_KEY_F && action == GLFW_PRESS) {
    dcel->visFill = !dcel->visFill;
  }
  if (key == GLFW_KEY_S && action == GLFW_PRESS) {
    dcel->visSkel = !dcel->visSkel;
  }
  if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
    dcel->useTessSM = !dcel->useTessSM;
    if (dcel->useTessSM) {
      if (dcel->useTessAltSM)
        printf("using full-SM tessVtx\n");
      else
        printf("using partial-SM tessVtx\n");
    } else {
      printf("using non-SM tessVtx\n");
    }
  }
  if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
    dcel->useTessAltSM = !dcel->useTessAltSM;
    if (dcel->useTessAltSM)
      printf("using full-SM tessVtx\n");
    else
      printf("using partial-SM tessVtx\n");
  }
  if (key == GLFW_KEY_3 && action == GLFW_PRESS) {
    dcel->useSampTex = !dcel->useSampTex;
    if (dcel->useSampTex)
      printf("using texture sampling patterns\n");
    else
      printf("using computed sampling patterns\n");
  }
  if (key == GLFW_KEY_4 && action == GLFW_PRESS) {
    dcel->useSvdUpdate = !dcel->useSvdUpdate;
    if (dcel->useSvdUpdate)
      printf("using rank-1 updating\n");
    else
      printf("using rank-n updating\n");
  }
  if (key == GLFW_KEY_5 && action == GLFW_PRESS) {
    dcel->useBlasUpdate = !dcel->useBlasUpdate;
    if (dcel->useBlasUpdate)
      printf("using cublas updating\n");
    else
      printf("using kernel updating\n");
  }

  if (key == GLFW_KEY_J && action == GLFW_PRESS) {
    phi += M_PI/18.0;
    updateCamera();
  }
  if (key == GLFW_KEY_L && action == GLFW_PRESS) {
    phi -= M_PI/18.0;
    updateCamera();
  }
  if (key == GLFW_KEY_K && action == GLFW_PRESS) {
    theta += M_PI/18.0;
    updateCamera();
  }
  if (key == GLFW_KEY_I && action == GLFW_PRESS) {
    theta -= M_PI/18.0;
    updateCamera();
  }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (leftMousePressed) {
    phi += (xpos - lastX) / xSize;
    theta -= (ypos - lastY) / ySize;
    updateCamera();
  }

  lastX = xpos;
  lastY = ypos;
}

void scrollCallback(GLFWwindow* window, double dx, double dy) {
  zoom += 30.0f*dy / ySize;
  zoom = std::fmax(0.01f, std::fmin(zoom, 100.0f));
  updateCamera();
}

void updateCamera() {
  theta = std::min((float)M_PI-0.001f, std::max(0.001f, theta));

  camPos.x = zoom * sin(phi) * sin(theta);
  camPos.y = zoom * cos(theta);
  camPos.z = -zoom * cos(phi) * sin(theta);


  projection = glm::perspective(fovy, float(xSize) / float(ySize), zNear, zFar);
  glm::mat4 view = glm::lookAt(camPos, glm::vec3(0,0,0), glm::vec3(0, 1, 0));
  projection = projection * view;

  shader->setUniform("model", projection);
}

