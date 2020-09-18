#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

class TrtObjectDetector{
public:
	TrtObjectDetector(const string filename) :
		engineFile(filename) {
			std::ifstream file(engineFile, std::ios::binary);
			vector<char> trtModelStreamFromFile;
			extractContentsToBuffer(file, trtModelStreamFromFile);
			Logger gLogger;
			// for plugin deserialization errors.
			initLibInferPlugins(gLogger, "");
			IRuntime* runtime_ = createInferRuntime(gLogger.getTRTLogger());
			assert(runtime_ != nullptr);
			mEngine = runtime_->deserializeCudaEngine(trtModelStreamFromFile.data(), size_engine); 
	}

	/**
		 * @brief function extracts content in .bin engine file to char vector
		 * @param file: ifstream object to which .bin engineFile is loaded in bin mode
		 * @param buf: char buffer to extracts contents of file
 */
	void extractContentsToBuffer(std::ifstream& file, std::vector <char>& buf) {
		if (file.good()) {
			file.seekg(0, file.end);
			size_engine = file.tellg();
			file.seekg(0, file.beg);
			buf.resize(size_engine);
			std::cout << "size of engine file: " << buf.size() << std::endl;
			file.read(buf.data(), size_engine);
			file.close();
		}
	}

	void doInference() {
		IExecutionContext* context = mEngine->createExecutionContext();
		assert(context != nullptr);
		int inputIndex = mEngine->getBindingIndex(INPUT_BLOB_NAME);
		int outputIndex = mEngine->getBindingIndex(OUTPUT_BLOB_NAME);

		void* buffersp[2];
	}


	private:
		std::string engineFile;
		std::size_t size_engine;

		std::unique_ptr<nvinfer1::ICudaEngine> mEngine;	
}

//! \class Logger
//!
//! \brief Class which manages logging of TensorRT tools and samples
//!
//! \details This class provides a common interface for TensorRT tools and samples to log information to the console,
//! and supports logging two types of messages:
//!
//! - Debugging messages with an associated severity (info, warning, error, or internal error/fatal)
//! - Test pass/fail messages

class Logger : public nvinfer1::ILogger {
public:
	Logger(Severity severity = Severity::kWARNING) : 
		mReportableSeverity(severity) {}

	//!
 //! \brief Forward-compatible method for retrieving the nvinfer::ILogger associated with this Logger
 //! \return The nvinfer1::ILogger associated with this Logger
	nvinfer1::ILogger& getTRTLogger() {
		return *this;
	}

	void log(Severity severity, const char* msg) override {
		//suppress info level messages
		if (severity != Severity::kINFO) {
			std::cout << msg << std::endl;
		}
	}

}

int main () {

}

