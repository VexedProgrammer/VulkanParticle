#include "VulkanEngine.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

VulkanEngine::VulkanEngine(VkPhysicalDevice & phyDevice, VkDevice & device) : m_PhyDevice(phyDevice), m_Device(device) {};

uint32_t VulkanEngine::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
	//Get properties of the gpu side memory
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(m_PhyDevice, &memProperties);

	//find and return the index of the suitable memory type for each memory type found in the memory properties
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanEngine::createBuffer(VkDeviceSize size, 
	VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer & buffer, VkDeviceMemory & bufferMemory, VkDescriptorBufferInfo* desc)
{

	//Set up generic buffer data
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = usage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	//Create buffer and error check
	if (vkCreateBuffer(m_Device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
		throw std::runtime_error("failed to create buffer!");
	}

	//Set up memory requirements for the buffer
	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(m_Device, buffer, &memRequirements);

	//Set up the allocation info using the meory requirements we just set up
	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

	//Allocate memory on the GPU and error check
	if (vkAllocateMemory(m_Device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate buffer memory!");
	}

	if (desc)
	{
		desc->buffer = buffer;
		desc->offset = 0;
		desc->range = VK_WHOLE_SIZE;
	}

	//Bind the buffer memory
	vkBindBufferMemory(m_Device, buffer, bufferMemory, 0);

}
#include <iostream>
void VulkanEngine::copyBuffer(VkQueue& graphicsQueue, VkCommandPool& comPool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands(comPool);

	VkBufferCopy copyRegion = {};
	copyRegion.size = size;
	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

	endSingleTimeCommands(graphicsQueue, comPool, commandBuffer);
}

VkCommandBuffer VulkanEngine::beginSingleTimeCommands(VkCommandPool& comPool)
{
	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = comPool;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(m_Device, &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	return commandBuffer;
}

void VulkanEngine::endSingleTimeCommands(VkQueue& graphicsQueue, VkCommandPool& comPool, VkCommandBuffer commandBuffer)
{
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(graphicsQueue);

	vkFreeCommandBuffers(m_Device, comPool, 1, &commandBuffer);
}

void VulkanEngine::createVertexBuffer(VkQueue& graphicsQueue, VkCommandPool& comPool, VulkanObject* object)
{
	//Calculate buffer size
	VkDeviceSize bufferSize = sizeof(Vertex) * object->GetVertices().size();


	//Use our create buffer function to get a generic staging buffer buffer
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory, nullptr);

	//Map the memory to a CPU side pointer and copy over the vertex data
	void* data;
	vkMapMemory(m_Device, stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, object->GetVertices().data(), (size_t)bufferSize);
	vkUnmapMemory(m_Device, stagingBufferMemory); //Unmap from cpu side

												  //Create a vertex buffer
	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, object->GetVertexBuffer(), object->GetVertexMemory(), nullptr);

	//Copy the staging buffer data to the vertex buffer
	copyBuffer(graphicsQueue, comPool, stagingBuffer, object->GetVertexBuffer(), bufferSize);

	//Destroy the staging buffer and free memory
	vkDestroyBuffer(m_Device, stagingBuffer, nullptr);
	vkFreeMemory(m_Device, stagingBufferMemory, nullptr);
}

void VulkanEngine::createIndexBuffer(VkQueue& graphicsQueue, VkCommandPool& comPool, VulkanObject* object)
{
	//Calculate buffer size
	VkDeviceSize bufferSize = sizeof(object->GetIndices()[0]) * object->GetIndices().size();

	//Set up staging buffer
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory, nullptr);

	//Map the data to  CPU side pointer and copy over the data, then unmap
	void* data;
	vkMapMemory(m_Device, stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, object->GetIndices().data(), (size_t)bufferSize);
	vkUnmapMemory(m_Device, stagingBufferMemory);

	//Create the index buffer
	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, object->GetIndexBuffer(), object->GetIndexMemory(), nullptr);

	//Copy the data from the staging buffer to the new index buffer
	copyBuffer(graphicsQueue, comPool, stagingBuffer, object->GetIndexBuffer(), bufferSize);


	//Clean up the staging buffer
	vkDestroyBuffer(m_Device, stagingBuffer, nullptr);
	vkFreeMemory(m_Device, stagingBufferMemory, nullptr);
}

void VulkanEngine::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage & image, VkDeviceMemory & imageMemory)
{
	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = width;
	imageInfo.extent.height = height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.format = format;
	imageInfo.tiling = tiling;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = usage;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateImage(m_Device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
		throw std::runtime_error("failed to create image!");
	}

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(m_Device, image, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

	if (vkAllocateMemory(m_Device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate image memory!");
	}

	vkBindImageMemory(m_Device, image, imageMemory, 0);
}

void VulkanEngine::createTextureImage(VkQueue& graphicsQueue, VkCommandPool& comPool, VkImage& textureImage, VkDeviceMemory& textureImageMemory, const char* texturePath)
{
	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load(texturePath, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	VkDeviceSize imageSize = texWidth * texHeight * 4;

	if (!pixels) {
		throw std::runtime_error("failed to load texture image!");
	}

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory, nullptr);

	void* data;
	vkMapMemory(m_Device, stagingBufferMemory, 0, imageSize, 0, &data);
	memcpy(data, pixels, static_cast<size_t>(imageSize));
	vkUnmapMemory(m_Device, stagingBufferMemory);

	stbi_image_free(pixels);

	createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

	transitionImageLayout(graphicsQueue, comPool, textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	copyBufferToImage(graphicsQueue, comPool, stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
	transitionImageLayout(graphicsQueue, comPool, textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	vkDestroyBuffer(m_Device, stagingBuffer, nullptr);
	vkFreeMemory(m_Device, stagingBufferMemory, nullptr);
}

void VulkanEngine::createNoiseTextureImage(VkQueue & graphicsQueue, VkCommandPool & comPool, VkImage & textureImage, VkDeviceMemory & textureImageMemory, float distribution)
{
	int texWidth = 256, texHeight = 256;
	//Random Noise
	std::uniform_real_distribution<float> randomFloats(0.0, 1.0); // random floats between 0.0 - 1.0
	std::default_random_engine generator;
	std::vector<glm::vec4> noiseArray;

	//For each pixel get a random value and if it is above the distribusion threshhold set it to white or if not, set it to the random grayscale colour
	for (unsigned int i = 0; i < texWidth*texHeight; i++)
	{
		float random = randomFloats(generator) * 2.0 - 1.0;
		if (random < distribution)
		{
			glm::vec4 noise(
				random,
				random,
				random,
				1.0f);
			noiseArray.push_back(noise);
		}
		else
		{
			glm::vec4 noise(
				0,
				0,
				0,
				0.0f);
			noiseArray.push_back(noise);
		}
	}
	VkDeviceSize imageSize = texWidth * texHeight * 4;

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory, nullptr);

	void* data;
	vkMapMemory(m_Device, stagingBufferMemory, 0, imageSize, 0, &data);
	memcpy(data, noiseArray.data(), static_cast<size_t>(imageSize));
	vkUnmapMemory(m_Device, stagingBufferMemory);


	createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

	transitionImageLayout(graphicsQueue, comPool, textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	copyBufferToImage(graphicsQueue, comPool, stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
	transitionImageLayout(graphicsQueue, comPool, textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	vkDestroyBuffer(m_Device, stagingBuffer, nullptr);
	vkFreeMemory(m_Device, stagingBufferMemory, nullptr);
}

void VulkanEngine::transitionImageLayout(VkQueue& graphicsQueue, VkCommandPool& comPool, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands(comPool);

	VkImageMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;

	if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

		if (hasStencilComponent(format)) {
			barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		}
	}
	else {
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	}

	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;

	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	}
	else {
		throw std::invalid_argument("unsupported layout transition!");
	}

	vkCmdPipelineBarrier(
		commandBuffer,
		sourceStage, destinationStage,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);

	endSingleTimeCommands(graphicsQueue, comPool, commandBuffer);
}

void VulkanEngine::copyBufferToImage(VkQueue& graphicsQueue, VkCommandPool& comPool, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands(comPool);

	VkBufferImageCopy region = {};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;
	region.imageOffset = { 0, 0, 0 };
	region.imageExtent = {
		width,
		height,
		1
	};

	vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	endSingleTimeCommands(graphicsQueue, comPool, commandBuffer);
}

void VulkanEngine::createTextureImageView(VulkanObject* object)
{
	object->SetTextureImageView(createImageView(object->GetTextureImage(), VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT));
}

void VulkanEngine::createTextureSampler(VulkanObject* object)
{
	VkSamplerCreateInfo samplerInfo = {};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_FALSE;
	samplerInfo.maxAnisotropy = 16;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	if (vkCreateSampler(m_Device, &samplerInfo, nullptr, &object->GetTextureSampler()) != VK_SUCCESS) {
		throw std::runtime_error("failed to create texture sampler!");
	}
}

VkImageView VulkanEngine::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
{
	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = aspectFlags;// VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	VkImageView imageView;
	if (vkCreateImageView(m_Device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
		throw std::runtime_error("failed to create texture image view!");
	}

	return imageView;
}


VkImageView VulkanEngine::createTextureImageView(VkImage& image)
{
	return createImageView(image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
}
void VulkanEngine::createTextureSampler(VkSampler& sampler)
{
	VkSamplerCreateInfo samplerInfo = {};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_FALSE;
	samplerInfo.maxAnisotropy = 16;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	if (vkCreateSampler(m_Device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
		throw std::runtime_error("failed to create texture sampler!");
	}
}






bool VulkanEngine::hasStencilComponent(VkFormat format)
{
	return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkResult VulkanEngine::createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer * buffer, VkDeviceMemory *memory, VkDeviceSize size, void * data)
{
	// Create the buffer handle
	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.usage = usageFlags;
	bufferCreateInfo.size = size;
	vkCreateBuffer(m_Device, &bufferCreateInfo, nullptr, buffer);

	// Create the memory backing up the buffer handle
	VkMemoryRequirements memReqs;
	VkMemoryAllocateInfo memAlloc{};
	memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	vkGetBufferMemoryRequirements(m_Device, *buffer, &memReqs);
	memAlloc.allocationSize = memReqs.size;
	// Find a memory type index that fits the properties of the buffer
	memAlloc.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags);
	vkAllocateMemory(m_Device, &memAlloc, nullptr, memory);

	// If a pointer to the buffer data has been passed, map the buffer and copy over the data
	if (data != nullptr)
	{
		void* map;
		vkMapMemory(m_Device, *memory, 0, size, 0, &map);
		memcpy(map, data, size);
		if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
		{
			VkMappedMemoryRange mappedRange = {};
			mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
			mappedRange.memory = *memory;
			mappedRange.offset = 0;
			mappedRange.size = size;
			vkFlushMappedMemoryRanges(m_Device, 1, &mappedRange);
		}

		vkUnmapMemory(m_Device, *memory);
	}


	// Attach the memory to the buffer object
	return vkBindBufferMemory(m_Device, *buffer, *memory, 0);
}


