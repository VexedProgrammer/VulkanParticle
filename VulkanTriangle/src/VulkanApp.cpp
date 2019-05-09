#include <VulkanApp.h>

static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {

	//Get the glfw window pointer cast to the vulkan app class, enable resizing
	auto app = reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(window));
	app->framebufferResized = true;
}


static std::vector<char> readFile(const std::string& filename) {

	//Open a file stream with the binary setting
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) { //Check if file opened
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t)file.tellg(); //Get file size
	std::vector<char> buffer(fileSize); //Allocate enough memory

	file.seekg(0); //Go to the start of the files
	file.read(buffer.data(), fileSize); //Read all data into the buffer

	file.close(); //close file stream

	return buffer; //Return character buffer
}

const void VulkanApp::initWindow()
{
	window = new GLFW_Window(512, 512, "Hello Pyramid"); //Open the GLFW window with a given size and name

	glfwSetWindowUserPointer(window->Window(), this); //Set the window pointer to this class (VulkanApp)
	glfwSetFramebufferSizeCallback(window->Window(), framebufferResizeCallback); //Set resize call back to given function
}

const void VulkanApp::initVulkan() {

	createInstance();
	setupDebugMessenger();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createSwapChain();
	createImageViews();
	createRenderPass();
	createDescriptorSetLayout();
	
	createCommandPool();

	//Creaate Objects after setting up required components

	/*m_Objects.push_back(new VulkanObject(m_Engine, physicalDevice, device, graphicsQueue, commandPool, "models/walls.obj", "textures/wall.jpg"));
	m_Objects[0]->SetPos(glm::vec3(-0.0f, -0.05f, 0));

	m_Objects.push_back(new VulkanObject(m_Engine, physicalDevice, device, graphicsQueue, commandPool, "models/sword.obj", "textures/sword.png"));
	m_Objects[1]->SetPos(glm::vec3(0.0f, -0.045f, 0));
	m_Objects[1]->SetRot(glm::vec3(0, 35.0f, 0));*/

	m_Engine->createTextureImage(graphicsQueue, commandPool, pTextureImage, pTextureImageMemory, "textures/particle.png");
	pTextureImageView = m_Engine->createTextureImageView(pTextureImage);
	m_Engine->createTextureSampler(pTextureSampler);
	pDescriptor.sampler = pTextureSampler;
	pDescriptor.imageView = pTextureImageView;
	pDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	m_Engine->createTextureImage(graphicsQueue, commandPool, gTextureImage, gTextureImageMemory, "textures/gradient.png");
	gTextureImageView = m_Engine->createTextureImageView(gTextureImage);
	m_Engine->createTextureSampler(gTextureSampler);
	gDescriptor.sampler = gTextureSampler;
	gDescriptor.imageView = gTextureImageView;
	gDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	
	
	createDepthResources();
	createFramebuffers();
	prepareStorageBuffers();
	createGraphicsPipeline();
	createUniformBuffers();
	createDescriptorPool();
	createDescriptorSets();
	prepareCompute();
	createCommandBuffers();
	createSyncObjects();

}

const void VulkanApp::createInstance()
{

	//Check if validation layers are supported, if not throw an error
	if (enableValidationLayers && !checkValidationLayerSupport()) {
		throw std::runtime_error("validation layers requested, but not available!");
	}

	//Set up app info, this is optional but useful
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "Hello FireWork";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "No Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	//Create instance info, used for getting GLFW extentions
	VkInstanceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;

	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;

	//Get required  instance extention
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	//Pass extention count and names to info
	createInfo.enabledExtensionCount = glfwExtensionCount;
	createInfo.ppEnabledExtensionNames = glfwExtensions;

	createInfo.enabledLayerCount = 0;

	//If using validation layers pass in validation data to info
	if (enableValidationLayers) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else {
		createInfo.enabledLayerCount = 0;
	}

	//Get required extention
	auto extensionsReq = getRequiredExtensions();
	createInfo.enabledExtensionCount = static_cast<uint32_t>(extensionsReq.size()); //Set data
	createInfo.ppEnabledExtensionNames = extensionsReq.data();

	//Create instance using set data/info
	VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

	//Check if the instance was created if note throw error
	if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
		throw std::runtime_error("failed to create instance!");
	}



	uint32_t extensionCount = 0;
	//Get extention proerties count
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
	std::vector<VkExtensionProperties> extensions(extensionCount);
	//Use count to actaully get the data
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());


	//Print out extention info to console
	std::cout << "available extensions:" << std::endl;

	for (const auto& extension : extensions) {
		std::cout << "\t" << extension.extensionName << std::endl;
	}

	m_Engine = new VulkanEngine(physicalDevice, device);
}

bool VulkanApp::checkValidationLayerSupport()
{

	uint32_t layerCount;
	//Get layer count
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
	//Allocate memory 
	std::vector<VkLayerProperties> availableLayers(layerCount);
	//Get player properties
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	//Go through each layer and check if it is supported return false is a required layer is not supported
	for (const char* layerName : validationLayers) {
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}

		if (!layerFound) {
			return false;
		}
	}

	return true;
}

const void VulkanApp::mainLoop() {
	//while window should not close keep looping
	while (!window->ShouldClose())
	{
		//Update window
		window->UpdateWindow();
		//Draw frame
		drawFrame();
	}
	//Wait for last frame to be processed before ending
	vkDeviceWaitIdle(device);
}

void VulkanApp::drawFrame() {
	//Compute
	auto tStart = std::chrono::high_resolution_clock::now();
	vkResetFences(device, 1, &compute.fence);
	VkSubmitInfo computeSubmitInfo{};
	computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	computeSubmitInfo.commandBufferCount = 1;
	computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;

	UpdateComputeUniform();

	vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, compute.fence);

	updateUniformBuffer(0, 0);
	//Wait for current frame to be processed before drawing a new one (stop memory leak)
	//vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

	//Get next image to render too, if failed recreate swap chain and wait till next frame
	uint32_t imageIndex;
	VkResult result = vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

	if (result == VK_ERROR_OUT_OF_DATE_KHR) {
		recreateSwapChain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("failed to acquire swap chain image!");
	}

	//for (unsigned int j = 0; j < m_Objects.size(); j++)
	//{
	//	//Update shader buffers
	//	updateUniformBuffer(imageIndex, j);
	//}

	////Set up submit info
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	//Pass in sync data (semiphores)
	VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;

	//Pass in command buffer data
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

	//Reset wait fence 
	//vkResetFences(device, 1, &inFlightFences[currentFrame]);


	//Submit data, graphics queue
	if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, 0) != VK_SUCCESS) {
		throw std::runtime_error("failed to submit draw command buffer!");
	}


	//Set up presentation settings
	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	VkSwapchainKHR swapChains[] = { swapChain };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;

	presentInfo.pImageIndices = &imageIndex;

	//Add info to queue
	result = vkQueuePresentKHR(presentQueue, &presentInfo);

	//Check if not valid or if the framebuffer has been resized, is not recreate swap chain
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
		framebufferResized = false;
		recreateSwapChain();
	}
	else if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to present swap chain image!");
	}

	//Update frame count
	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;



	
	vkWaitForFences(device, 1, &compute.fence, VK_TRUE, UINT64_MAX);
	auto tEnd = std::chrono::high_resolution_clock::now();
	auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
	frameTimer = (float)tDiff / 1000.0f;
}



const void VulkanApp::cleanup() {

	//Clean up memory from swap chain
	cleanupSwapChain();

	//Clean up descipter pool memory
	vkDestroyDescriptorPool(device, descriptorPool, nullptr);

	//Clean up layout memory
	vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
	//Clean up shader buffers
	/*for (size_t i = 0; i < swapChainImages.size(); i++) {
		for (unsigned int j = 0; j < m_Objects.size(); j++)
		{
			unsigned int index = m_Objects.size() * i + j;
			vkDestroyBuffer(device, uniformBuffers[index], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[index], nullptr);
		}
	}*/

	/*for (unsigned int i = 0; m_Objects.size(); i++)
	{
		delete m_Objects[i];
	}
*/
	vkDestroyRenderPass(device, renderPass, nullptr); //Clean up render pass data

	//Clean up semaphore/sync objects
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
		vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
		vkDestroyFence(device, inFlightFences[i], nullptr);
	}

	//clean up command pools
	vkDestroyCommandPool(device, commandPool, nullptr);

	//Clean up device
	vkDestroyDevice(device, nullptr);


	//Clean up debugging
	if (enableValidationLayers) {
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
	}

	//Clean up instance.surface
	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyInstance(instance, nullptr);

	//Clean up glfw window
	delete window;
}

std::vector<const char*> VulkanApp::getRequiredExtensions() {

	//Get extention count
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	//Allocate vector memory
	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	//Add extention names too vector
	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	//Return vector
	return extensions;
}


VKAPI_ATTR VkBool32 VKAPI_CALL VulkanApp::debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {

	//Return error message from validation layer
	std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

	return VK_FALSE;
}

void VulkanApp::setupDebugMessenger() {
	
	//Ignore validation if disabled
	if (!enableValidationLayers) return;

	//Set debugging flags
	VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback; //Set call back function

	//Create debug messenger, error is the creation fails
	if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
		throw std::runtime_error("failed to set up debug messenger!");
	}

}

VkResult VulkanApp::CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	

	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void VulkanApp::DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	
	//Get destroy proxy function for cleaning up the debug messenger
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

void VulkanApp::pickPhysicalDevice() {

	//Get physical device count
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

	//If 0 error, no vulkan enabled GPUs!
	if (deviceCount == 0) {
		throw std::runtime_error("failed to find GPUs with Vulkan support!");
	}

	//Allocate enough memory to store device data and pass data into the vector
	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

	//Loop through devices and look for a suitable device, just use first device found for now
	for (const auto& device : devices) {
		if (isDeviceSuitable(device)) {
			physicalDevice = device;
			break;
		}
	}

	//If no divices are suitable, throw and error
	if (physicalDevice == VK_NULL_HANDLE) {
		throw std::runtime_error("failed to find a suitable GPU!");
	}

	//Print device count
	std::cout << "Count: " << deviceCount << " " << physicalDevice << std::endl;
}

bool VulkanApp::isDeviceSuitable(VkPhysicalDevice device) {

	//Check which queue families are support to check that all available commands are available
	QueueFamilyIndices indices = findQueueFamilies(device);

	//Check tha all extentions needed are supported
	bool extensionsSupported = checkDeviceExtensionSupport(device);

	//Make sure the swap chain is supported on the device
	bool swapChainAdequate = false;
	if (extensionsSupported) {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
		swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}

	VkPhysicalDeviceFeatures supportedFeatures;
	vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

	//Only return true if all conditions are met
	return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}

bool VulkanApp::checkDeviceExtensionSupport(VkPhysicalDevice device) {

	//Get extention count
	uint32_t extensionCount;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

	//Allocate vector memory and get extention data
	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

	//Get a vector of all required extentions
	std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

	//Go through and remove extention names in teh vector
	for (const auto& extension : availableExtensions) {
		requiredExtensions.erase(extension.extensionName);
	}
	
	//If all the devices names are erased then all are supported, return true
	return requiredExtensions.empty();
}

VulkanApp::QueueFamilyIndices VulkanApp::findQueueFamilies(VkPhysicalDevice device) {
	QueueFamilyIndices indices;

	//Get queue count
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

	//Allocate memory and store family queue data
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

	//Loop through all retreived queues
	int i = 0;
	for (const auto& queueFamily : queueFamilies) {

		//Check for graphics flag and update graphics family
		if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			indices.graphicsFamily = i;
		}
		//Check if the support is present
		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

		//IF support is found and queue count is greater than 0 update presentFamily
		if (queueFamily.queueCount > 0 && presentSupport) {
			indices.presentFamily = i;
		}

		//If all data is complete break the loop
		if (indices.isComplete()) {
			break;
		}

		i++;
	}

	return indices;
}

void VulkanApp::createLogicalDevice() {

	//Get queue family data
	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

	//Allocate memory and set the graphics and present data
	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

	//Loop through each family and set up queue info and add them to the vector
	float queuePriority = 1.0f;
	for (uint32_t queueFamily : uniqueQueueFamilies) {
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}



	VkPhysicalDeviceFeatures deviceFeatures = {};

	deviceFeatures.wideLines = VK_TRUE;

	//Set up logical device info
	VkDeviceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

	//Pass queue data
	createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	createInfo.pQueueCreateInfos = queueCreateInfos.data();

	//Set Enabled devices features
	createInfo.pEnabledFeatures = &deviceFeatures;

	//Set extentions
	createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
	createInfo.ppEnabledExtensionNames = deviceExtensions.data();


	//Set up validation layers for logical device if enabled
	if (enableValidationLayers) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else {
		createInfo.enabledLayerCount = 0;
	}

	//Create deivce and check to make sure it was created, otherwise throw and error
	if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
		throw std::runtime_error("failed to create logical device!");
	}

	vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
	vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
}

void VulkanApp::createSurface() {

	//Create window rendering surface using vulkan instance infomation 
	if (glfwCreateWindowSurface(instance, window->Window(), nullptr, &surface) != VK_SUCCESS) {
		throw std::runtime_error("failed to create window surface!");
	}
}

VulkanApp::SwapChainSupportDetails VulkanApp::querySwapChainSupport(VkPhysicalDevice device) {
	
	//Query the the basic surface capabilities
	SwapChainSupportDetails details;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	//Get the supported surface formats
	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

	if (formatCount != 0) {
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
	}

	//Query the supported presentation modes
	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

	if (presentModeCount != 0) {
		details.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
	}

	//Return swap chain details
	return details;
}

VkSurfaceFormatKHR VulkanApp::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
	
	//In the case that the surface has no prefered formats, just choose one to use
	if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
		return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
	}

	//Otherwise we need to check if the preferred combinations is available
	for (const auto& availableFormat : availableFormats) {
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return availableFormat;
		}
	}

	return availableFormats[0];
}

VkPresentModeKHR VulkanApp::chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes) {

	//Set best mode to VK_PRESENT_MODE_FIFO_KHR as it is garenteed to be available
	VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

	//Loop through and check for VK_PRESENT_MODE_MAILBOX_KHR for triple buffering
	//for (const auto& availablePresentMode : availablePresentModes) {
	//	if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
	//		return availablePresentMode;
	//	}//Otherwise if not supported, try to get the basic mode for displaying images straight away
	//	else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
	//		bestMode = availablePresentMode;
	//	}
	//}

	return bestMode;
}

VkExtent2D VulkanApp::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
	
	//If the current resolution is max, keep the current resolution
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		return capabilities.currentExtent;
	}
	else { //Otherwise query the framebuffer size from glfw to return the window resolution
		int width, height;
		glfwGetFramebufferSize(window->Window(), &width, &height);

		VkExtent2D actualExtent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		return actualExtent;
	}
}

void VulkanApp::createSwapChain() {

	//Check for swap chain support data
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

	//Set up the formats from the swap chain details
	VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
	VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
	VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

	//Request more than the minimum images to make sure we dont stall waiting for another image to render too
	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

	//Make sure we don't exceed the maximum image count
	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	//Set up swap chain info struct
	VkSwapchainCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = surface; //Give it the rendering surface

	createInfo.minImageCount = imageCount; //Swapchain image count
	createInfo.imageFormat = surfaceFormat.format; 
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent; //Resolution
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	//Get and set up the queue families supported on the selected physical device
	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
	uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

	if (indices.graphicsFamily != indices.presentFamily) {
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else {
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	//USe current transform, we don't want to transform the resulting images at all
	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; //Set up alpha blending
	createInfo.presentMode = presentMode; //Pass in resent mode
	createInfo.clipped = VK_TRUE; //Clip pixels that are obscured

	//For now only ever create one swap chain
	createInfo.oldSwapchain = VK_NULL_HANDLE;


	//Create swap chain and error check
	if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
		throw std::runtime_error("failed to create swap chain!");
	}

	//Set up memory for the swap chain images and store the inital data
	vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
	swapChainImages.resize(imageCount);
	vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

	swapChainImageFormat = surfaceFormat.format;
	swapChainExtent = extent;
}

void VulkanApp::createImageViews() {
	
	//Allocate enough memory for each image
	swapChainImageViews.resize(swapChainImages.size());

	//For each image set up the info struct and create the image view
	for (uint32_t i = 0; i < swapChainImages.size(); i++) {
		swapChainImageViews[i] = m_Engine->createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
	}

}

void VulkanApp::createGraphicsPipeline() {

	//Read in shader files
	//Base mesh and Shell rendering
	auto vertShaderCode = readFile("shaders/vert.spv");
	auto fragShaderCode = readFile("shaders/frag.spv");


	//Set up shader modules for both vertex and fragment shaders
	VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
	VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

	//Set up vertex info
	VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; //Vertex stage
	vertShaderStageInfo.module = vertShaderModule; //Vertex shader
	vertShaderStageInfo.pName = "main"; //Main function as entry point

	//Set up fragment info
	VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT; //Fragment stage
	fragShaderStageInfo.module = fragShaderModule; //Fragment shader
	fragShaderStageInfo.pName = "main"; //Main function as entry point

	//List shader stage info in order
	VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

	//Set up vertex input pipline info
	VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

	//Get descriptions
	auto bindingDescription = vertices.bindingDescriptions; //Vertex::getBindingDescription();
	auto attributeDescriptions = vertices.attributeDescriptions; //Vertex::getAttributeDescriptions();

	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data(); //Give binding desc
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data(); //Give attribute data

	//Assembly state info (rendering type)
	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; //Rendering using triangle lists
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	//Set up viewport
	viewport = {};
	viewport.x = 0.0f; //No offset
	viewport.y = 0.0f;//No offset
	//Resolution
	viewport.width = (float)swapChainExtent.width; 
	viewport.height = (float)swapChainExtent.height;
	//Depth buffer range
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	//No scissor ( dont want to cut any of the image out)
	VkRect2D scissor = {};
	scissor.offset = { 0, 0 };
	scissor.extent = swapChainExtent;

	//Viewport info
	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	//Rasterizer info
	VkPipelineRasterizationStateCreateInfo rasterizer = {};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE; //Dont clamp depth
	rasterizer.rasterizerDiscardEnable = VK_FALSE; //Dont discard (wont draw to framebuffer otherwise)
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL; //We want to draw full polygons not lines or points
	rasterizer.lineWidth = 1.0f; 
	rasterizer.cullMode = VK_CULL_MODE_NONE; //Cull back faces
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; //Set clockwise 
	rasterizer.depthBiasEnable = VK_FALSE;

	//Default multisamplign settings (disabled for now)
	VkPipelineMultisampleStateCreateInfo multisampling = {};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineDepthStencilStateCreateInfo depthStencil = {};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_FALSE;
	depthStencil.depthWriteEnable = VK_FALSE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.stencilTestEnable = VK_FALSE;
	//depthStencil.minDepthBounds = 0.0f; // Optional
	//depthStencil.maxDepthBounds = 1.0f; // Optional

	//Set up colour blending settings (Default blending settings)
	VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_TRUE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

	VkPipelineColorBlendStateCreateInfo colorBlending = {};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	//Layout info (mainly default)
	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

	std::vector<VkDynamicState> dynamicStateEnables = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
	};
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.pDynamicStates = dynamicStateEnables.data();
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());
	dynamicState.flags = 0;

	//Create layout and error check
	if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create pipeline layout!");
	}

	//Set up graphics pipline info
	VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2; //2 shader stages
	//Pass in all data set up before this
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertices.inputState;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = renderPass;
	pipelineInfo.subpass = 0;
	//pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; //Only going to be using one pipline for now, dont need to refrence the base
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pDynamicState = &dynamicState;

	VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
	pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache);

	//Create pipeline and error check
	if (vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
		throw std::runtime_error("failed to create graphics pipeline!");
	}
	

	//Destroy shader modules now we have finished with them
	vkDestroyShaderModule(device, fragShaderModule, nullptr);
	vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

VkShaderModule VulkanApp::createShaderModule(const std::vector<char>& code) {

	//Set up up shader module info
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size(); //Pass in shader size
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); //Pass in shader code

	//Create shader module and error check
	VkShaderModule shaderModule;
	if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
		throw std::runtime_error("failed to create shader module!");
	}

	return shaderModule;
}

void VulkanApp::createRenderPass() {
	

	VkAttachmentDescription colorAttachment = {};
	colorAttachment.format = swapChainImageFormat;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; //Colour attachment must match swap chain format
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; //Clear after frame
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; //Store frame-buffer after render so we can use it later
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; //Not using stencil buffer for this
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; //Not using stencil buffer for this
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;  //Ignore previos layout
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //Use image in the swap buffer

	VkAttachmentDescription depthAttachment = {};
	depthAttachment.format = findDepthFormat();
	depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	//Refrence to the attachment for the sub passes
	VkAttachmentReference colorAttachmentRef = {};
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachmentRef = {};
	depthAttachmentRef.attachment = 1;
	depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	//Sub pass info
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;
	subpass.pDepthStencilAttachment = &depthAttachmentRef;
	subpass.inputAttachmentCount = 0;
	subpass.pInputAttachments = nullptr;
	subpass.preserveAttachmentCount = 0;
	subpass.pPreserveAttachments = nullptr;
	subpass.pResolveAttachments = nullptr;

	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;


	std::array<VkSubpassDescription, 1> subpasses = { subpass };
	std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
	//Set up the final render pass info passing in the above data
	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
	renderPassInfo.pAttachments = attachments.data();
	renderPassInfo.subpassCount = static_cast<uint32_t>(subpasses.size());
	renderPassInfo.pSubpasses = subpasses.data();
	renderPassInfo.dependencyCount = dependencies.size();
	renderPassInfo.pDependencies = dependencies.data();

	//Create render pass and error check
	if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
		throw std::runtime_error("failed to create render pass!");
	}
}

void VulkanApp::createFramebuffers() {

	//Allocate memory
	swapChainFramebuffers.resize(swapChainImageViews.size());

	//For each swap chain view create a frame buffer
	for (size_t i = 0; i < swapChainImageViews.size(); i++) {
		std::array<VkImageView, 2> attachments = {
				swapChainImageViews[i],
				depthImageView
		};

		VkFramebufferCreateInfo framebufferInfo = {};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass; //Pass in the render pass
		framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());; //Two attachments in the swap chain (include depth)
		framebufferInfo.pAttachments = attachments.data(); //pass in the swap chain image view as an attachment along with depth/stencil
		framebufferInfo.width = swapChainExtent.width; //Resolution
		framebufferInfo.height = swapChainExtent.height;
		framebufferInfo.layers = 1; //Just 1 layer

		//Create frame buffer and error check
		if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create framebuffer!");
		}
	}
}

void VulkanApp::createCommandPool() {

	//Get the queue family so we can referance the graphics family
	QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(); //Pass in the graphics family value

	//Create command pool and error check
	if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create command pool!");
	}
}

void VulkanApp::createCommandBuffers() {
	
	//Allocate memory
	commandBuffers.resize(swapChainFramebuffers.size());

	//Set up command buffer info
	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = commandPool; //Pass in the command pool to be part of
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = (uint32_t)commandBuffers.size(); //Number of command buffers

	//Allocate command buffers
	if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate command buffers!");
	}

	//For each command buffer set up info and bind the required data
	for (size_t i = 0; i < commandBuffers.size(); i++) {
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		//beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

		if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}


		std::array<VkClearValue, 2> clearValues = {};
		clearValues[0].color = { 0.2f, 0.2f, 0.2f, 1.0f };//Set clear colour
		clearValues[1].depthStencil = { 1.0f, 0 };  

		VkRenderPassBeginInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass; //Pass renderpass
		renderPassInfo.framebuffer = swapChainFramebuffers[i]; //Pass frame buffer
		renderPassInfo.renderArea.offset = { 0, 0 }; //No offset
		renderPassInfo.renderArea.extent = swapChainExtent; //Set resolution
		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size()); //Clear value to 1
		renderPassInfo.pClearValues = clearValues.data(); //Pass in clear colour




		//Begin render pass so we can bind
		


		

		//Set up and bind in vertex infomation
		std::vector<VkBuffer> vertexBuffers;
		/*for (unsigned int v = 0; v < m_Objects.size(); v++)
		{
			vertexBuffers.push_back(m_Objects[v]->GetVertexBuffer());
		}*/

		//VkBuffer vertexBuffers[] = { m_Objects[j]->GetVertexBuffer() };
		VkDeviceSize offsets[] = { 0 };		

		// Draw the particle system using the update vertex buffer

		vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdSetViewport(commandBuffers[i], 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.extent.width = swapChainExtent.width;
		scissor.extent.height = swapChainExtent.height;
		scissor.offset.x = 0;
		scissor.offset.y = 0;
		vkCmdSetScissor(commandBuffers[i], 0, 1, &scissor);

		vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
		vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[0], 0, NULL);

	
		vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, &compute.storageBuffer, offsets);
		vkCmdDraw(commandBuffers[i], PARTICLE_COUNT, 1, 0, 0);


		vkCmdEndRenderPass(commandBuffers[i]);


		//Check the command has ended and error check
		if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}
}

void VulkanApp::createSyncObjects()
{

	//Allocate memory for all sync objects
	imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

	//Set up semaphore info
	VkSemaphoreCreateInfo semaphoreInfo = {};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;


	//Set up fence info
	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;


	//For each frame in flight (buffering) clear availabe and finsihed semiforces as well as wait fences
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
			vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
			vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
			//Throw error if any creation fails
			throw std::runtime_error("failed to create synchronization objects for a frame!");
		}
	}
}

void VulkanApp::recreateSwapChain() {
	
	//Get the new width and height if either are set to 0
	int width = 0, height = 0;
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(window->Window(), &width, &height);
		window->setSize(glm::vec2(width, height));
		glfwWaitEvents();
	}
	//Wait for end of frame
	vkDeviceWaitIdle(device);

	//Delete current swapchain data
	cleanupSwapChain();


	//Create all parts of the swap chain
	createSwapChain();
	createImageViews();
	createRenderPass();
	createGraphicsPipeline();
	createDepthResources();
	createFramebuffers();
	createCommandBuffers();
}

void VulkanApp::cleanupSwapChain() {

	vkDestroyImageView(device, depthImageView, nullptr);
	vkDestroyImage(device, depthImage, nullptr);
	vkFreeMemory(device, depthImageMemory, nullptr);

	//Destroy all frame buffers
	for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
		vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
	}
	//free memory from command buffers
	vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

	//Destroy graphics pipline and layout
	vkDestroyPipeline(device, graphicsPipeline, nullptr);
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

	//Destroy all image views
	for (size_t i = 0; i < swapChainImageViews.size(); i++) {
		vkDestroyImageView(device, swapChainImageViews[i], nullptr);
	}
	
	//Destroy the actual swapchain object
	vkDestroySwapchainKHR(device, swapChain, nullptr);
}




void VulkanApp::createDescriptorSetLayout()
{
	/*VkDescriptorSetLayoutBinding uboLayoutBinding = {};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	uboLayoutBinding.pImmutableSamplers = nullptr;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
	samplerLayoutBinding.binding = 1;
	samplerLayoutBinding.descriptorCount = 1;
	samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	samplerLayoutBinding.pImmutableSamplers = nullptr;
	samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutBinding depthSamplerLayoutBinding = {};
	depthSamplerLayoutBinding.binding = 2;
	depthSamplerLayoutBinding.descriptorCount = 1;
	depthSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	depthSamplerLayoutBinding.pImmutableSamplers = nullptr;
	depthSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;*/
	

	VkDescriptorSetLayoutBinding uboLayoutBinding = {};
	uboLayoutBinding.binding = 2;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.pImmutableSamplers = nullptr;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	VkDescriptorSetLayoutBinding colour = {};
	colour.binding = 0;
	colour.descriptorCount = 1;
	colour.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	colour.pImmutableSamplers = nullptr;
	colour.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	VkDescriptorSetLayoutBinding gradient = {};
	gradient.binding = 1;
	gradient.descriptorCount = 1;
	gradient.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	gradient.pImmutableSamplers = nullptr;
	gradient.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	//std::array<VkDescriptorSetLayoutBinding, 3> bindings = { uboLayoutBinding, samplerLayoutBinding, depthSamplerLayoutBinding };// guboLayoutBinding
	std::array<VkDescriptorSetLayoutBinding, 3> bindings = { uboLayoutBinding, colour, gradient };
	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();

	if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor set layout!");
	}

}

void VulkanApp::createUniformBuffers()
{
	//Get size of buffer
	VkDeviceSize bufferSize = sizeof(UniformBufferObject);

	unsigned int size = 0;
	/*for (unsigned int i = 0; i < m_Objects.size(); i++)
	{
		size += swapChainImages.size();
	}*/

	//Allocate memory
	uniformBuffers.resize(1);
	uniformBuffersMemory.resize(1);

	//Create a uniform buffer for each of the swap chain images
	for (size_t i = 0; i < 1; i++) {
		m_Engine->createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i], nullptr);
	}

	m_Engine->createBuffer(sizeof(compute.ubo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, compute.uniformBuffer, compute.uniformMemory, &compute.uniformDesc);
}

void VulkanApp::updateUniformBuffer(uint32_t currentImage, unsigned int objectIndex)
{
	float x = 1;
	float y = 0;
	//theta += 1 * frameTimer;
	float x_rotated = ((x - 0) * cos(theta)) - ((0 - y) * sin(theta)) + 0;
	float y_rotated = ((0 - y) * cos(theta)) - ((x - 0) * sin(theta)) + 0;
	//Set up the uniform model matrix (rotation and translation and scale)
	UniformBufferObject ubo = {};
	ubo.model = glm::identity<glm::mat4>();
	//ubo.model = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0)) * glm::rotate(ubo.model, glm::radians(0.0f), glm::vec3(0, 1, 0)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.05f, 0.05f, 0.05f));
	
	
	//View matrix using look at
	ubo.view =  glm::lookAt(glm::vec3(x_rotated, 0.1f, 15), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	//Projection / Perspective matrix
	glm::mat4 proj = glm::perspective(glm::radians(67.0f), (float)swapChainExtent.width / (float)swapChainExtent.height, 0.01f, 100.0f);
	proj[1][1] *= -1;
	ubo.proj = proj;
	//Map memory to a CPU side pointer, copy over data then unmap from cpu side
	
	void* data;
	vkMapMemory(device, uniformBuffersMemory[0], 0, sizeof(ubo), 0, &data);
	memcpy(data, &ubo, sizeof(ubo));
	vkUnmapMemory(device, uniformBuffersMemory[0]);
}

void VulkanApp::createDescriptorPool()
{
	unsigned int size = 0;
	/*for (unsigned int i = 0; i < m_Objects.size(); i++)
	{
		size += swapChainImages.size();
	}*/

	/*std::array<VkDescriptorPoolSize, 3> poolSizes = {};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizes[0].descriptorCount = static_cast<uint32_t>(size*4);
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[1].descriptorCount = static_cast<uint32_t>(size*4);
	poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[2].descriptorCount = static_cast<uint32_t>(size * 4);*/

	std::array<VkDescriptorPoolSize, 3> poolSizes = {};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizes[0].descriptorCount = static_cast<uint32_t>(2);
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[1].descriptorCount = static_cast<uint32_t>(2);
	poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[2].descriptorCount = static_cast<uint32_t>(1);


	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = 2;// static_cast<uint32_t>(size * 4);

	if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor pool!");
	}
}

void VulkanApp::createDescriptorSets()
{
	descriptorSets.resize(1);

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.pSetLayouts = &descriptorSetLayout;
	allocInfo.descriptorSetCount = 1;

	vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets[0]);

	std::vector<VkWriteDescriptorSet> writeDescriptorSets;
	// Binding 0 : Particle color map
	VkWriteDescriptorSet writeDescriptorSet{};
	writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.dstSet = descriptorSets[0];
	writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writeDescriptorSet.dstBinding = 0;
	writeDescriptorSet.pImageInfo = &pDescriptor;
	writeDescriptorSet.descriptorCount = 1;
	writeDescriptorSets.push_back(writeDescriptorSet);
	// Binding 1 : Particle gradient ramp
	VkWriteDescriptorSet writeDescriptorSetTwo{};
	writeDescriptorSetTwo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSetTwo.dstSet = descriptorSets[0];
	writeDescriptorSetTwo.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writeDescriptorSetTwo.dstBinding = 1;
	writeDescriptorSetTwo.pImageInfo = &gDescriptor;
	writeDescriptorSetTwo.descriptorCount = 1;
	writeDescriptorSets.push_back(writeDescriptorSetTwo);
	// Binding : UBO
	VkDescriptorBufferInfo bufferInfo = {};
	bufferInfo.buffer = uniformBuffers[0]; //Actual buffer to use
	bufferInfo.offset = 0; //Start at the start
	bufferInfo.range = sizeof(UniformBufferObject); //Size of each buffer
	VkWriteDescriptorSet writeDescriptorSetUBO{};
	writeDescriptorSetUBO.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSetUBO.dstSet = descriptorSets[0];
	writeDescriptorSetUBO.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writeDescriptorSetUBO.dstBinding = 2;
	writeDescriptorSetUBO.pBufferInfo = &bufferInfo;
	writeDescriptorSetUBO.descriptorCount = 1;
	writeDescriptorSets.push_back(writeDescriptorSetUBO);

	vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
}

void VulkanApp::createDepthResources()
{
	VkFormat depthFormat = findDepthFormat();

	m_Engine->createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
	depthImageView = m_Engine->createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

	m_Engine->transitionImageLayout(graphicsQueue, commandPool, depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

VkFormat VulkanApp::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
	for (VkFormat format : candidates) {
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

		if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
			return format;
		}
		else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
			return format;
		}
	}

	throw std::runtime_error("failed to find supported format!");
}

VkFormat VulkanApp::findDepthFormat()
{
	return findSupportedFormat(
		{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
	);
}

void VulkanApp::prepareStorageBuffers()
{
	std::default_random_engine rndEngine(true ? 0 : (unsigned)time(nullptr));
	std::uniform_real_distribution<float> rndDist(-0.001f, 0.001f);
	std::uniform_real_distribution<float> rndDist01(0.2f, 1.0f);
	std::uniform_real_distribution<float> rndDist10(-30.0f, 30.0f);
	
	// Initial particle positions
	std::vector<Particle> particleBuffer(PARTICLE_COUNT);
	for (auto& particle : particleBuffer) {
		float r = rndDist01(rndEngine);
		float r10 = rndDist10(rndEngine);
		particle.pos =  glm::vec4(0, 15, 0, 0);
		particle.vel = glm::normalize(glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 0))*(400*r);
		particle.startvel = particle.vel;
		particle.gradientPos.x = r;
		particle.lifeTimer.y = r*1.5f;
		particle.lifeTimer.z = r10;
		particle.lifeTimer.a = r10;
		particle.rootPos = glm::vec4(0, -5, 0, 0);
	}

	VkDeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);

	// Staging
	// SSBO won't be changed on the host after upload so copy to device local memory 

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;

	m_Engine->createBuffer(
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&stagingBuffer,
		&stagingMemory,
		storageBufferSize,
		particleBuffer.data());

	

	m_Engine->createBuffer(
		// The SSBO will be used as a storage buffer for the compute pipeline and as a vertex buffer in the graphics pipeline
		storageBufferSize,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		compute.storageBuffer,
		compute.storageMemory,
		&compute.storageDesc);



	// Copy to staging buffer
	VkCommandBuffer copyCmd;
	VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
	commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	commandBufferAllocateInfo.commandPool = commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 1;
	
	vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &copyCmd);

	// If requested, also start recording for the new command buffer
	if (true)
	{
		VkCommandBufferBeginInfo cmdBufferBeginInfo{};
		cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		vkBeginCommandBuffer(copyCmd, &cmdBufferBeginInfo);
	}

	VkBufferCopy copyRegion = {};
	copyRegion.size = storageBufferSize;
	vkCmdCopyBuffer(copyCmd, stagingBuffer, compute.storageBuffer, 1, &copyRegion);
	flushCommandBuffer(copyCmd, graphicsQueue, true);

	vkDestroyBuffer(device, stagingBuffer, nullptr);
	vkFreeMemory(device, stagingMemory, nullptr);

	// Binding description
	vertices.bindingDescriptions.resize(1);
	VkVertexInputBindingDescription vInputBindDescription{};
	vInputBindDescription.binding = 0;
	vInputBindDescription.stride = sizeof(Particle);
	vInputBindDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	vertices.bindingDescriptions[0] = vInputBindDescription;
		
	

	// Attribute descriptions
	// Describes memory layout and shader positions
	vertices.attributeDescriptions.resize(2);
	// Location 0 : Position
	VkVertexInputAttributeDescription vInputAttribDescription{};
	vInputAttribDescription.location = 0;
	vInputAttribDescription.binding = 0;
	vInputAttribDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
	vInputAttribDescription.offset = offsetof(Particle, pos);
	vertices.attributeDescriptions[0] = vInputAttribDescription;
	
	VkVertexInputAttributeDescription vInputAttribDescriptionTwo{};
	vInputAttribDescriptionTwo.location = 1;
	vInputAttribDescriptionTwo.binding = 0;
	vInputAttribDescriptionTwo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	vInputAttribDescriptionTwo.offset = offsetof(Particle, gradientPos);
	// Location 1 : Gradient position
	vertices.attributeDescriptions[1] = vInputAttribDescriptionTwo;
	

	// Assign to vertex buffer
	VkPipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo{};
	pipelineVertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertices.inputState = pipelineVertexInputStateCreateInfo;
	vertices.inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertices.bindingDescriptions.size());
	vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
	vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.attributeDescriptions.size());
	vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
}

void VulkanApp::flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free)
{
	if (commandBuffer == VK_NULL_HANDLE)
	{
		return;
	}

	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	// Create fence to ensure that the command buffer has finished executing
	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = 0;
	VkFence fence;
	vkCreateFence(device, &fenceInfo, nullptr, &fence);

	// Submit to the queue
	vkQueueSubmit(queue, 1, &submitInfo, fence);
	// Wait for the fence to signal that command buffer has finished executing
	vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);

	vkDestroyFence(device, fence, nullptr);

	if (free)
	{
		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}
}

void VulkanApp::prepareCompute()
{
	// Create a compute capable device queue
		// The VulkanDevice::createLogicalDevice functions finds a compute capable queue and prefers queue families that only support compute
		// Depending on the implementation this may result in different queue family indices for graphics and computes,
		// requiring proper synchronization (see the memory barriers in buildComputeCommandBuffer)
	vkGetDeviceQueue(device, computeFamily, 0, &compute.queue);

	// Create compute pipeline
	// Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)

	VkDescriptorSetLayoutBinding setLayoutBinding{};
	setLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	setLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	setLayoutBinding.binding = 0;
	setLayoutBinding.descriptorCount = 1;

	VkDescriptorSetLayoutBinding setLayoutBindingTwo{};
	setLayoutBindingTwo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	setLayoutBindingTwo.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	setLayoutBindingTwo.binding = 1;
	setLayoutBindingTwo.descriptorCount = 1;
	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
		//Particle position storage buffer
		setLayoutBinding,
		//Uniform buffer
		setLayoutBindingTwo
	};

	VkDescriptorSetLayoutCreateInfo descriptorLayout{};
	descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorLayout.pBindings = setLayoutBindings.data();
	descriptorLayout.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
	
	vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout);

	VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo{};
	pPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pPipelineLayoutCreateInfo.setLayoutCount = 1;
	pPipelineLayoutCreateInfo.pSetLayouts = &compute.descriptorSetLayout;

	vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout);

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.pSetLayouts = &compute.descriptorSetLayout;
	allocInfo.descriptorSetCount = 1;

	vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet);

	VkWriteDescriptorSet writeDescriptorSet{};
	writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.dstSet = compute.descriptorSet;
	writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writeDescriptorSet.dstBinding = 0;
	writeDescriptorSet.pBufferInfo = &compute.storageDesc;
	writeDescriptorSet.descriptorCount = 1;
	VkWriteDescriptorSet writeDescriptorSetTwo{};
	writeDescriptorSetTwo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSetTwo.dstSet = compute.descriptorSet;
	writeDescriptorSetTwo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writeDescriptorSetTwo.dstBinding = 1;
	writeDescriptorSetTwo.pBufferInfo = &compute.uniformDesc;
	writeDescriptorSetTwo.descriptorCount = 1;
	std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets =
	{
		//Particle position storage buffer
		writeDescriptorSet,
		//Uniform buffer
		writeDescriptorSetTwo
	};

	vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

	// Create pipeline	
	VkComputePipelineCreateInfo computePipelineCreateInfo{};
	computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCreateInfo.layout = compute.pipelineLayout;
	computePipelineCreateInfo.flags = 0;


	auto compShaderCode = readFile("shaders/comp.spv");
	//Set up shader modules for both vertex and fragment shaders
	VkShaderModule compShaderModule = createShaderModule(compShaderCode);
	VkPipelineShaderStageCreateInfo shaderStage = {};
	shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStage.pName = "main"; 
	shaderStage.module = compShaderModule;
	//Set up vertex info


	computePipelineCreateInfo.stage = shaderStage;
	vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline);

	// Separate command pool as queue family for compute may be different than graphics
	VkCommandPoolCreateInfo cmdPoolInfo = {};
	cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmdPoolInfo.queueFamilyIndex = computeFamily;
	cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool);

	// Create a command buffer for compute operations
	VkCommandBufferAllocateInfo cmdBufAllocateInfo{};
	cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdBufAllocateInfo.commandPool = compute.commandPool;
	cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdBufAllocateInfo.commandBufferCount = 1;

	vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &compute.commandBuffer);

	// Fence for compute CB sync
	VkFenceCreateInfo fenceCreateInfo{};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
	vkCreateFence(device, &fenceCreateInfo, nullptr, &compute.fence);

	// Build a single command buffer containing the compute dispatch commands
	buildComputeCommandBuffer();
}

void VulkanApp::buildComputeCommandBuffer()
{
	VkCommandBufferBeginInfo cmdBufInfo{};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo);

	// Compute particle movement

	// Add memory barrier to ensure that the (graphics) vertex shader has fetched attributes before compute starts to write to the buffer
	VkBufferMemoryBarrier bufferBarrier{};
	bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

	bufferBarrier.buffer = compute.storageBuffer;
	bufferBarrier.size = compute.storageDesc.range;
	bufferBarrier.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;						// Vertex shader invocations have finished reading from the buffer
	bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;								// Compute shader wants to write to the buffer
	// Compute and graphics queue may have different queue families (see VulkanDevice::createLogicalDevice)
	// For the barrier to work across different queues, we need to set their family indices

	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
	bufferBarrier.srcQueueFamilyIndex = indices.graphicsFamily.value();			// Required as compute and graphics queue may have different families
	bufferBarrier.dstQueueFamilyIndex = computeFamily;			// Required as compute and graphics queue may have different families

	vkCmdPipelineBarrier(
		compute.commandBuffer,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0,
		0, nullptr,
		1, &bufferBarrier,
		0, nullptr);

	vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);
	vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);

	// Dispatch the compute job
	vkCmdDispatch(compute.commandBuffer, (PARTICLE_COUNT / 256), 1, 1);

	// Add memory barrier to ensure that compute shader has finished writing to the buffer
	// Without this the (rendering) vertex shader may display incomplete results (partial data from last frame) 
	bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;								// Compute shader has finished writes to the buffer
	bufferBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;						// Vertex shader invocations want to read from the buffer
	bufferBarrier.buffer = compute.storageBuffer;
	bufferBarrier.size = compute.storageDesc.range;
	// Compute and graphics queue may have different queue families (see VulkanDevice::createLogicalDevice)
	// For the barrier to work across different queues, we need to set their family indices
	bufferBarrier.srcQueueFamilyIndex = computeFamily;			// Required as compute and graphics queue may have different families
	bufferBarrier.dstQueueFamilyIndex = indices.graphicsFamily.value();			// Required as compute and graphics queue may have different families

	vkCmdPipelineBarrier(
		compute.commandBuffer,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
		0,
		0, nullptr,
		1, &bufferBarrier,
		0, nullptr);

	vkEndCommandBuffer(compute.commandBuffer);
}

void VulkanApp::UpdateComputeUniform()
{
	timer += frameTimer;
	if (timer > 10.0f) timer = 0.0f;
	compute.ubo.deltaT = frameTimer * 2.5f;
	compute.ubo.destX = sin(glm::radians(timer * 360.0f)) * 0.75f;
	compute.ubo.destY = 0.0f;
	compute.ubo.time = timer;

	void* data;
	vkMapMemory(device, compute.uniformMemory, 0, sizeof(compute.ubo), 0, &data);
	memcpy(data, &compute.ubo, sizeof(compute.ubo));
	vkUnmapMemory(device, compute.uniformMemory);
}

