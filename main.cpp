#include <cstdio>
#include <cmath>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "glm/gtc/matrix_transform.hpp"

#include "dcel.hpp"
#include "shader.hpp"

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
Shader *shader, *ppmShader;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("missing arg\n");
    return 0;
  }

  // initialize window
  if(!glfwInit()) {
    printf("glfw err\n");
    exit(EXIT_FAILURE);
  }
  GLFWwindow* window = glfwCreateWindow(xSize, ySize, "Scene Graph", NULL, NULL);
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

  //glEnable(GL_BLEND);
  //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glfwSetWindowSizeCallback(window, resizeCallback);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetScrollCallback(window, scrollCallback);

  //create the dcel
  DCEL dcel(argv[1]);
  printf("created dcel\n");

  //create our shader
  shader = new Shader();
  shader->setShader("dcel.frag.glsl", GL_FRAGMENT_SHADER);
  shader->setShader("dcel.vert.glsl", GL_VERTEX_SHADER);
  shader->recompile();
  printGLErrorLog();

  //create our tesselation shader
  ppmShader = new Shader();
  /*
  ppmShader->setShader("ppm.tcl.glsl", GL_TESS_CONTROL_SHADER);
  ppmShader->setShader("ppm.tel.glsl", GL_TESS_EVALUATION_SHADER);
  ppmShader->setShader("ppm.frag.glsl", GL_FRAGMENT_SHADER);
  ppmShader->setShader("ppm.vert.glsl", GL_VERTEX_SHADER);
  ppmShader->recompile();
  printGLErrorLog();
  */

  printf("begin main loop\n");
  int nbFrames = 0;
  double lastTime = glfwGetTime();
  updateCamera();
  double dt = 0, alpha = .95;
  while(!glfwWindowShouldClose(window)) {
    // Clear the screen so that we only see newly drawn images
    // glfwSetCursorPos(window, xSize/2, ySize/2);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* TODO: draw things here */
    double t1 = clock();
    dcel.visUpdate();
    dcel.visDraw(shader, ppmShader);
    double t2 = clock();
    dt = (t2-t1)*(1.0-alpha) + dt*alpha;

    // Move the rendering we just made onto the screen
    glfwSwapBuffers(window);
    glfwPollEvents();

    double currentTime = glfwGetTime();
    nbFrames++;
    if (currentTime - lastTime >= 1.0){
       printf("%.1f fps (dt = %f us)\n", double(nbFrames)/(currentTime - lastTime), 1.0E6*dt/CLOCKS_PER_SEC);
       nbFrames = 0;
       lastTime += 1.0;
    }

    // Check for any GL errors that have happened recently
    printGLErrorLog();
  }

  glfwDestroyWindow(window);
  glfwTerminate();
  delete shader;
  delete ppmShader;

  return 0;
}

void resizeCallback(GLFWwindow* window, int x, int y)
{
  xSize = x;
  ySize = y;
  glViewport(0,0,x,y);
  updateCamera();
}

void errorCallback(int error, const char *description) {
  fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (leftMousePressed) {
    // compute new camera parameters
    phi += (xpos - lastX) / xSize;
    theta -= (ypos - lastY) / ySize;
    theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
    updateCamera();
  }

  lastX = xpos;
  lastY = ypos;
}

void scrollCallback(GLFWwindow* window, double dx, double dy) {
  zoom += 10.0f*dy / ySize;
  zoom = std::fmax(0.1f, std::fmin(zoom, 10.0f));
  updateCamera();
}

void updateCamera() {
  camPos.x = zoom * sin(phi) * sin(theta);
  camPos.z = zoom * cos(theta);
  camPos.y = zoom * cos(phi) * sin(theta);


  projection = glm::perspective(fovy, float(xSize) / float(ySize), zNear, zFar);
  glm::mat4 view = glm::lookAt(camPos, glm::vec3(0,0,0), glm::vec3(0, 0, 1));
  projection = projection * view;

  shader->setUniform("model", projection);
}
