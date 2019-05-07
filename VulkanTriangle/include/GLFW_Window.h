#pragma once

#define GLFW_INCLUDE_VULKAN
#include <glfw3.h>

#include <iostream>
#include <GLM\vec2.hpp>

class GLFW_Window
{
private:
	GLFWwindow * window;
	uint32_t extensionCount = 0;

	unsigned int WIDTH, HIEGHT;

public:
	GLFW_Window(unsigned int width, unsigned int height, const char* title);
	~GLFW_Window();

	void UpdateWindow();

	const bool ShouldClose() const { return glfwWindowShouldClose(window); }

	GLFWwindow* Window() const { return window; }

	glm::vec2 getSize() const { return glm::vec2(WIDTH, HIEGHT); }
	const void setSize(glm::vec2 size) { WIDTH = size.x; HIEGHT = size.y; }
};