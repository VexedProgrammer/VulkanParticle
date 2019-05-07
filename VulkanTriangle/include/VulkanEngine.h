#pragma once

#define GLFW_INCLUDE_VULKAN
#include <glfw3.h>

#include <cstdlib>
#include <stdexcept>

#include <vector>

#include <GLM/glm.hpp>
#include <GLM/gtc/matrix_transform.hpp>
#include "VulkanObject.h"
#include <random>

class VulkanEngine
{

private:
	VkPhysicalDevice& m_PhyDevice;
	VkDevice& m_Device;
public: 
	VulkanEngine(VkPhysicalDevice& phyDevice, VkDevice& device);
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	VkCommandBuffer beginSingleTimeCommands(VkCommandPool& comPool);
	void endSingleTimeCommands(VkQueue& graphicsQueue, VkCommandPool& comPool, VkCommandBuffer commandBuffer);
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory, VkDescriptorBufferInfo* desc);
	void copyBuffer(VkQueue& graphicsQueue, VkCommandPool& comPool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

	void createVertexBuffer(VkQueue& graphicsQueue, VkCommandPool& comPool, VulkanObject* object);
	void createIndexBuffer(VkQueue& graphicsQueue, VkCommandPool& comPool, VulkanObject* object);

	//Textures
	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
	void createTextureImage(VkQueue& graphicsQueue, VkCommandPool& comPool, VkImage& textureImage, VkDeviceMemory& textureImageMemory, const char* texturePath);
	void createNoiseTextureImage(VkQueue& graphicsQueue, VkCommandPool& comPool, VkImage& textureImage, VkDeviceMemory& textureImageMemory, float distribution);
	void transitionImageLayout(VkQueue& graphicsQueue, VkCommandPool& comPool, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
	void copyBufferToImage(VkQueue& graphicsQueue, VkCommandPool& comPool, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

	void createTextureImageView(VulkanObject* object);
	void createTextureSampler(VulkanObject* object);
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);

	VkImageView createTextureImageView(VkImage& image);
	void createTextureSampler(VkSampler& sampler);

	bool hasStencilComponent(VkFormat format);

	VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size, void *data = nullptr);
};