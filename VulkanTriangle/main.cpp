
#include <VulkanApp.h>
#include <iostream>
int main() {
	//Create app
	VulkanApp* app = new VulkanApp();

	//Try to run the app or exit with an error
	try {
		app->run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}