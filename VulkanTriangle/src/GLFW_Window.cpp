#include "GLFW_Window.h"

GLFW_Window::GLFW_Window(unsigned int width, unsigned int height, const char * title)
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(width, height, title, nullptr, nullptr);
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

	WIDTH = width;
	HIEGHT = height;

	std::cout << extensionCount << " extensions supported" << std::endl;
}

GLFW_Window::~GLFW_Window()
{
	glfwDestroyWindow(window);
	glfwTerminate();
}

void GLFW_Window::UpdateWindow()
{
	glfwPollEvents();
}

