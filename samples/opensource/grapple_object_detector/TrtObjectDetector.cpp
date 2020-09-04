#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

class TrtObjectDetector{
	public:
	 TrtObjectDetector(const std::string filename) : 
	 	engineFile(filename) {
			 std::ifstream file(engineFile, std::ios::binary);
			 vector<char> trtModelStreamFromFile;
			 extractContentsToBuffer(file, trtModelStreamFromFile);
			 //TODO: steal the logging class from 
			 //TensorRT/samples/common/logging.h
			 IRuntime* runtime_ = createInferRuntime(gLogger.getTRTLogger());
			 engine_ = runtime_->deserializeCudaEngine(trtModelStreamfromFile.data(),
																									size, 
																									nullptr);
			
		}

	private:
		string engineFile;
		// TensorRT engine used to run the network
		std::unique_ptr<nvinfer1::ICudaEngine> mEngine;

		/**
		 * @brief function extracts content in .bin engine file to char vector
		 * @param file: ifstream object to which .bin engineFile is loaded in bin mode
		 * @param buf: char buffer to extracts contents of file
		 */
		void extractContentsToBuffer(std::ifstream& file, 
																std::vector <char>& buf) {
			size_t size{0};
			if (file.good()) {
				file.seekg(0, file.end);
				size = file.tellg();
				file.seekg(0, file.beg);
				trtModelStreamFromFile.resize(size);
				file.read(trtModelStreamFromFile.data(), size);
				file.close();
			}
		}

		


}

