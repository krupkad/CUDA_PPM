#include "shader.hpp"

#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <stdexcept>

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"

static std::string textFileRead(std::string filename);
static void printLinkInfoLog(int prog);
static void printShaderInfoLog(int shader);

Shader* Shader::inUse = nullptr;

Shader::Shader() :  hasProgram(false) {}

Shader::~Shader() {
  for (const auto &itr : shMap)
    glDeleteShader(itr.second);
  if(hasProgram)
    glDeleteProgram(program);
}

Shader::Shader(std::string src) : hasProgram(false)  {
  setShader(src + ".vert.glsl", GL_VERTEX_SHADER);
  setShader(src + ".frag.glsl", GL_FRAGMENT_SHADER);
  setShader(src + ".geom.glsl", GL_GEOMETRY_SHADER);
  recompile();
}

GLuint Shader::setShader(const std::string &src, GLenum type) {
  const auto itr = shMap.find(type);
  if (itr != shMap.end())
    glDeleteShader(itr->second);

  std::string sourceStr = textFileRead(src);
  const char *sourceCStr = sourceStr.c_str();
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &sourceCStr, NULL);
  glCompileShader(shader);
  printShaderInfoLog(shader);
  shMap[type] = shader;
  return shader;
}

void Shader::setUniform(std::string name, const glm::vec3& v) {
  int loc = resolveUniform(name);
  if(loc < 0)
    return;
  glUniform3fv(loc, 1, glm::value_ptr(v));
}

void Shader::setUniform(std::string name, float x, float y, float z) {
  int loc = resolveUniform(name);
  if(loc < 0)
    return;
  glUniform3f(loc, x, y, z);
}

void Shader::setUniform(std::string name, bool v) {
  int loc = resolveUniform(name);
  if(loc < 0)
    return;
  glUniform1i(loc, v);
}

void Shader::setUniform(std::string name, int v) {
  int loc = resolveUniform(name);
  if(loc < 0)
    return;
  glUniform1i(loc, v);
}

void Shader::setUniform(std::string name, const glm::mat4& m) {
  int loc = resolveUniform(name);
  if(loc < 0)
    return;
  glUniformMatrix4fv(loc, 1, GL_FALSE, &m[0][0]);
}

void Shader::bindVertexData(std::string name, unsigned int vbo) {
  int loc = resolveAttribute(name);
  if(loc < 0)
    return;
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(loc, 3, GL_FLOAT, false, 0, NULL);
  glEnableVertexAttribArray(loc);
}


void Shader::bindVertexData(std::string name, unsigned int vbo, int sso) {
  int loc = resolveAttribute(name);
  if(loc < 0)
    return;
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  int size = SHADER_SSO_SIZE(sso);
  int stride = SHADER_SSO_STRIDE(sso) * sizeof(float);
  void *offset = (void*)(SHADER_SSO_OFFSET(sso) * sizeof(float));
  glVertexAttribPointer(loc, size, GL_FLOAT, false, stride, offset);
  glEnableVertexAttribArray(loc);
}

void Shader::bindIndexData(unsigned int vbo) {
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
}

void Shader::unbindVertexData(std::string name) {
  int loc = resolveAttribute(name);
  if(loc < 0)
    return;
  glDisableVertexAttribArray(loc);
}

void Shader::recompile() {
  if(hasProgram)
    glDeleteProgram(program);
  program = glCreateProgram();

  for (const auto &itr : shMap)
    glAttachShader(program,itr.second);
  glLinkProgram(program);
  printLinkInfoLog(program);
  for (const auto &itr : shMap)
    glDetachShader(program,itr.second);

  hasProgram = true;
}

int Shader::resolveUniform(std::string name) {
  use();

  auto itr = uMap.find(name);
  int loc;
  if(itr != uMap.end()) {
    loc = itr->second;
  } else {
    loc = glGetUniformLocation(program, name.c_str());
    if(loc < 0)
      std::cerr << "warning: couldn't resolve uniform '" << name << "', not binding...\n";
    uMap[name] = loc;
  }

  return loc;
}

int Shader::resolveAttribute(std::string name) {
  use();

  auto itr = aMap.find(name);
  int loc;
  if(itr != aMap.end()) {
    loc = itr->second;
  } else {
    loc = glGetAttribLocation(program, name.c_str());
    if(loc < 0)
      std::cerr << "warning: couldn't resolve attribute '" << name << "', not binding...\n";
    aMap[name] = loc;
  }

  return loc;
}

static std::string textFileRead(std::string filename) {
  std::ifstream in(filename.c_str());
  if (!in) {
      std::cerr << "Error reading file " << filename << std::endl;
      throw (errno);
  }
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static void printLinkInfoLog(int prog) {
  GLint linked;
  glGetProgramiv(prog, GL_LINK_STATUS, &linked);
  if (linked == GL_TRUE) {
      return;
  }
  std::cerr << "GLSL LINK ERROR" << std::endl;

  int infoLogLen = 0;
  int charsWritten = 0;
  GLchar *infoLog;

  glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infoLogLen);

  if (infoLogLen > 0) {
      infoLog = new GLchar[infoLogLen];
      // error check for fail to allocate memory omitted
      glGetProgramInfoLog(prog, infoLogLen, &charsWritten, infoLog);
      std::cerr << "InfoLog:" << std::endl << infoLog << std::endl;
      delete[] infoLog;
  }
  // Throwing here allows us to use the debugger to track down the error.
  throw;
}

static void printShaderInfoLog(int shader) {
  GLint compiled;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (compiled == GL_TRUE) {
      return;
  }
  std::cerr << "GLSL COMPILE ERROR" << std::endl;

  int infoLogLen = 0;
  int charsWritten = 0;
  GLchar *infoLog;

  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);

  if (infoLogLen > 0) {
      infoLog = new GLchar[infoLogLen];
      // error check for fail to allocate memory omitted
      glGetShaderInfoLog(shader, infoLogLen, &charsWritten, infoLog);
      std::cerr << "InfoLog:" << std::endl << infoLog << std::endl;
      delete[] infoLog;
  }
  // Throwing here allows us to use the debugger to track down the error.
  throw;
}


void printGLErrorLog()
{
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error " << error << ": ";
        const char *e =
            error == GL_INVALID_OPERATION             ? "GL_INVALID_OPERATION" :
            error == GL_INVALID_ENUM                  ? "GL_INVALID_ENUM" :
            error == GL_INVALID_VALUE                 ? "GL_INVALID_VALUE" :
            error == GL_INVALID_INDEX                 ? "GL_INVALID_INDEX" :
            "unknown";
        std::cerr << e << std::endl;

        // Throwing here allows us to use the debugger stack trace to track
        // down the error.
#ifndef __APPLE__
        // But don't do this on OS X. It might cause a premature crash.
        // http://lists.apple.com/archives/mac-opengl/2012/Jul/msg00038.html
        throw;
#endif
    }
}

void Shader::use() {
  if(hasProgram) {
    glUseProgram(program);
    inUse = this;
  }
}

void Shader::requireAttr(std::string name) {
  int l = resolveAttribute(name);
  if(l < 0) {
    std::string w = "shader missing required attribute '" + name + "'";
    throw std::runtime_error(w.c_str());
  }
}

void Shader::requireAttr(std::string name, int loc) {
  int l = resolveAttribute(name);
  if(l < 0 || l != loc) {
    std::string w = "shader missing required attribute '" + name + "'";
    throw std::runtime_error(w.c_str());
  }
}

void Shader::requireUniform(std::string name) {
  int l = resolveUniform(name);
  if(l < 0) {
    std::string w = "shader missing required uniform '" + name + "'";
    throw std::runtime_error(w.c_str());
  }
}

void Shader::requireUniform(std::string name, int loc) {
  int l = resolveUniform(name);
  if(l < 0 || l != loc) {
    std::string w = "shader missing required uniform '" + name + "'";
    throw std::runtime_error(w.c_str());
  }
}
