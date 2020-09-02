#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

std::ofstream p(engineFile.c_str(), std::ios::binary);
IHostMemory* m_ModelStream = engine->serialize();
p.write(reinterpret_cast<const char*>(m_ModelStream->data()), m_ModelStream->size());
p.close();

std::vector<char> trtModelStreamfromFile;	
size_t size{ 0 };
std::ifstream file(engine_file, std::ios::binary);
if (file.good()) {
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
	trtModelStreamfromFile.resize(size);
	file.read(trtModelStreamfromFile.data(), size);
	file.close();
	//IRuntime* infer = createInferRuntime(gLogger.getTRTLogger());
	engine_ = runtime_->deserializeCudaEngine(trtModelStreamfromFile.data(), size, nullptr);
}

class TrtObjectDetector{
	public:
	 TrtObjectDetector(const std::string filename) : 
	 	engineFile(filename) {
			 vector<char> trtModelStreamFromFile;
			 size_t size{0};
			 std::ifstream file(engineFile, std::ios::binary);
			 if (file.good()) {
				 file.seekg(0, file.end);
				 size = file.tellg();
				 file.seekg(0, file.beg);
				 trtModelStreamFromFile.resize(size);
				 file.read(trtModelStreamFromFile.data(), size);
				 file.close();
			}
		}

	private:
		string engineFile;
		// TensorRT engine used to run the network
		std::unique_ptr<nvinfer1::ICudaEngine> mEngine; 
		


}

