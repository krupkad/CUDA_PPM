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
void updateCamera();
int xSize = 640, ySize = 480;
float lastX, lastY, xpos, ypos;
bool leftMousePressed = false, rightMousePressed = false;
float fovy = (float) (M_PI / 4), zNear = 0.10f, zFar = 100.0f;
float theta = 1.22f, phi = -0.70f, zoom = 10.0f;
glm::vec3 camPos;
glm::mat4 projection;
Shader *shader;

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
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
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

  //create the dcel
  DCEL dcel(argv[1]);
  printf("created dcel\n");

  //create our shader
  shader = new Shader();
  shader->setFragShader("dcel.frag.glsl");
  shader->setVertShader("dcel.vert.glsl");
  shader->recompile();
  printGLErrorLog();

  printf("begin main loop\n");
  int nbFrames = 0;
  double lastTime = glfwGetTime();
  updateCamera();
  while(!glfwWindowShouldClose(window)) {
    // Clear the screen so that we only see newly drawn images
    // glfwSetCursorPos(window, xSize/2, ySize/2);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* TODO: draw things here */
    dcel.visDraw(shader);

    // Move the rendering we just made onto the screen
    glfwSwapBuffers(window);
    glfwPollEvents();

    double currentTime = glfwGetTime();
    nbFrames++;
    if (currentTime - lastTime >= 1.0){
       printf("%.1f fps\n", double(nbFrames)/(currentTime - lastTime));
       nbFrames = 0;
       lastTime += 1.0;
    }

    // Check for any GL errors that have happened recently
    printGLErrorLog();
  }

  glfwDestroyWindow(window);
  glfwTerminate();
  delete shader;

  return 0;
}

void resizeCallback(GLFWwindow* window, int x, int y)
{
  xSize = x;
  ySize = y;
  //scene->resize(xSize,ySize);
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
  else if (rightMousePressed) {
    zoom += (ypos - lastY) / ySize;
    zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
    updateCamera();
  }

  lastX = xpos;
  lastY = ypos;
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
