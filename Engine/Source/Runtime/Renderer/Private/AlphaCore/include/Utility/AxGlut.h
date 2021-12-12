#ifndef __ALPHA_CORE_GLUT_H__
#define __ALPHA_CORE_GLUT_H__

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
namespace AlphaUtility
{
	namespace GL
	{
		static void CreateOpenGLView(int argc, char **argv, int width, int height, const char* title)
		{
			glutInit(&argc, argv);
			glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
			glutInitWindowPosition(100, 100);
			glutInitWindowSize(width, height);
			glutCreateWindow(title);

			// default initialization
			glClearColor(0.18, 0.18, 0.18, 0.0);
			glEnable(GL_DEPTH_TEST);
		}
	}
}

#endif // !__ALPHA_CORE_GLUT_H__
