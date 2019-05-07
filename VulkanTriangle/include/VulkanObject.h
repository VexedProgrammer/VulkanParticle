#pragma once




#define GLFW_INCLUDE_VULKAN
#include <glfw3.h>

#include <GLM/glm.hpp>
#include <GLM/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <array>
#include <vector>


#include <unordered_map>




class VulkanEngine;

/*! Vertex Struct
Holds the vertex information, main the position and colour.
*/
struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 texCoord;

	//Returns the vertex binding desciption
	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription = {};
		bindingDescription.binding = 0; //Bind to 0
		bindingDescription.stride = sizeof(Vertex); //Size of the vertex struct for each vertex
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; //Input as a vertex

		return bindingDescription;
	}

	//Get the details for each attribute stream
	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

		//Both are bound to zero as they are both in the same stream
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0; //Location at 0
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT; //Vector3D/RGB value
		attributeDescriptions[0].offset = offsetof(Vertex, pos); //Offset upto the vertex position (current this is no offset)

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1; //Location 1
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT; //Vector3D
		attributeDescriptions[1].offset = offsetof(Vertex, normal); //Offset upto the normal variable in the vertex struct

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}
	bool operator==(const Vertex& other) const {
		return pos == other.pos && normal == other.normal && texCoord == other.texCoord;
	}
};
namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

class VulkanObject
{
private:

	glm::vec3 m_Position = glm::vec3(0, 0, 0);
	glm::vec3 m_Rotation = glm::vec3(0, 0, 0);

	VulkanEngine* m_Engine;
	VkPhysicalDevice& m_PhyDevice;
	VkDevice& m_Device;
	VkQueue m_GraphicsPipline;
	VkCommandPool m_CommandPool;
	

	//Textures
	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	VkSampler textureSampler;

	

	//Vector of vertacies each with a position and colour
	std::vector<Vertex> vertices; /*= {
		{ { -1.0f, -1.0f, -1.0f },{ -1.0f, -1.0f, -1.0f }, {0.0f, 0.3f} },
		{ { 1.0f, -1.0f, -1.0f },{ 1.0f, -1.0f, -1.0f } , {0.5f, 1.0f}  },
		{ { 1.0f, -1.0f,  1.0f },{ 1.0f, -1.0f,  1.0f } , {0.4f, 0.1f}  },
		{ { -1.0f, -1.0f,  1.0f },{ -1.0f, -1.0f,  1.0f }, {0.85f, 0.0f} },
		{ { -0.0f,  1.0f,  0.0f },{ 0.0f,  1.0f,  0.0f }, {1.0f, 0.0f} }
	};*/
	//Index of each vertex in order to make the pyrimid shapes
	std::vector<uint32_t> indices; /*= {
		2, 0, 1, 0, 2, 3, 2, 1, 4, 0, 3, 4, 3, 2, 4, 1, 0, 4
	};*/

	//Vertex Buffers
	VkBuffer m_VertexBuffer;
	VkDeviceMemory m_VertexBufferMemory;
	VkBuffer m_IndexBuffer;
	VkDeviceMemory m_IndexBufferMemory;

public:

	VulkanObject(VulkanEngine* engine, VkPhysicalDevice& phyDevice, VkDevice& device, VkQueue graphicsQueue, VkCommandPool commandPool, const char* modelPath, const char* texturePath);
	~VulkanObject();

	

	VkBuffer& GetVertexBuffer() { return m_VertexBuffer; }
	VkBuffer& GetIndexBuffer() { return m_IndexBuffer; }
	const std::vector<uint32_t>& GetIndices() { return indices; }
	const std::vector<Vertex>& GetVertices() { return vertices; }
	VkDeviceMemory& GetVertexMemory() { return m_VertexBufferMemory; }
	VkDeviceMemory& GetIndexMemory() { return m_IndexBufferMemory; }

	const void SetPos(glm::vec3 pos) { m_Position = pos; }
	const glm::vec3 GetPos() const { return m_Position; }

	const void SetRot(glm::vec3 rot) { m_Rotation = rot; }
	const glm::vec3 GetRot() const { return m_Rotation; }

	VkImage& GetTextureImage() { return textureImage; }
	void SetTextureImageView(VkImageView view) { textureImageView = view; }
	VkImageView& GetTextureImageView() { return textureImageView; }
	VkSampler& GetTextureSampler() { return textureSampler; }

	void loadModel(const char* path);

};