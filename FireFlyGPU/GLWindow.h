#pragma once
#include <fstream>
#include <Windows.h>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include <iostream>
#include "GlobalSetting.h"

extern "C" void renderFrame();
//extern "C" void RefreshFps(int value);

namespace GLWindow
{
	GLint bufferSize;
	GLbyte* buffer;
	const GLint rgbwidth = Width * 4;

	//void Render();

	inline void Resize(int width, int height)
	{
		glutReshapeWindow(Width, Height);
	}

	inline void WindowsUpdate()
	{
		renderFrame();

		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(Width, Height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
		glutSwapBuffers();
		glFlush();
	}

	inline void TimerProc(int id)
	{
		glutPostRedisplay();
		glutTimerFunc(1, TimerProc, 1);
	}

	inline void InitWindow(int argc, char** argv, unsigned int mode, int x_position, int y_position, int width, int heigth, const char * title)
	{
		glutInit(&argc, argv);
		glutInitDisplayMode(mode);
		glutInitWindowPosition(x_position, y_position);
		glutInitWindowSize(width, heigth);
		glutCreateWindow(title);
		glClearColor(1.0, 0.0, 1.0, 1.0);
		glMatrixMode(GL_PROJECTION);
		gluOrtho2D(0.0, 200.0, 0.0, 150.0);

		//glutTimerFunc(1000, RefreshFps, 1000);

		glutDisplayFunc(WindowsUpdate);
		glutTimerFunc(1, TimerProc, 1);
		glutReshapeFunc(Resize);
		glutMainLoop();
	}


	inline void Savepic()
	{
		std::ofstream outf;
		outf.open("/Output/abc.ppm");
		outf << "P3\n" << Width << " " << Height << "\n255\n";
		for (auto h = Height - 1; h >= 0; h--)
		{
			for (int i = 0; i < rgbwidth; i += 3)
			{
				outf << buffer[h *(rgbwidth)+(i + 0)] << " " <<
					buffer[h *(rgbwidth)+(i + 1)] << " " <<
					buffer[h *(rgbwidth)+(i + 2)] << " \n";
			}
		}
		outf.close();
		std::cout << "finished" << std::endl;
	}
}