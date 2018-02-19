#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <CL/cl.hpp>
#include "Utils.h"

using namespace std;

vector<int> temperature;
int mean;
long totalOperatingTime;

void print_help() {
	cerr << "Application usage:" << endl;

	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -h : print this message" << endl;
}

void readData(string file)
{
	string line;
	ifstream myFile(file);
	string inputData = "";
	char space = ' ';
	int counter = 0;
	int temp = 0;
	//Check if file is open
	if (myFile.is_open())
	{
		cout << "Reading file...\n";
		//Gets line from file
		while (getline(myFile, line))
		{
			//Loop through each character in line
			for (int i = 0; i < line.length(); i++)
			{
				//Add char[i] to string inputData
				inputData += line[i];
				//If char is space or end of line
				if (line[i] == space || i == line.length() - 1)
				{
					counter++;
					switch (counter)
					{
						//Get every 6th value which is temperature
					case 6:
						temp = int(stof(inputData) * 10);
						temperature.push_back(temp);
						counter = 0;
						inputData = "";
						break;

					default:
						inputData = "";
						break;
					}
				}
			}
		}
		myFile.close();
	}
	else
	{
		cout << "Unable to open file" << endl;
		exit(0);
	}
}

void average(cl::CommandQueue queue, cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, size_t local_size, size_t vector_size, size_t vector_elements)
{
	//Create output vector and output size
	vector<int> output(1);
	size_t output_size = output.size() * sizeof(int);

	//Sets the kernel and the arguments for the kernel
	cl::Kernel kernel = cl::Kernel(program, "avg");
	kernel.setArg(0, buffer_A);
	kernel.setArg(1, buffer_B);
	kernel.setArg(2, cl::Local(local_size * sizeof(int)));

	//Copy to device memory
	cl::Event write_event;
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &temperature[0], NULL, &write_event);
	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

	//Runs the kernel with a thread for each element of the input with workgroups sized the same as local size
	cl::Event queue_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size), NULL, &queue_event);

	//Reads the output from GPU to CPU
	cl::Event read_event;
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &output[0], NULL, &read_event);

	//Calculate time to perform memory transfer and kernel execution
	//Time to write to memory
	long memoryWrite = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Time to read from memory
	long memoryRead = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Total time of reading and writing to memory
	long memoryExecution = memoryWrite + memoryRead;
	//Time for kernel to execute
	long kernelExecution = queue_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - queue_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Total of memory exectution and kernel execution time
	long operatingTime = memoryExecution + kernelExecution;

	//Calculates and prints Average
	float answer = ((float)output[0] / (float)10) / (float)vector_elements;
	cout << "Average: " << answer << endl;
	cout << "Operating time: " << operatingTime << "ns" << endl;
	totalOperatingTime += operatingTime;
	mean = answer * 10;
}

void maxTemp(cl::CommandQueue queue, cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, size_t local_size, size_t vector_size, size_t vector_elements)
{
	//Create output vector and output size
	vector<int> output(1);
	size_t output_size = output.size() * sizeof(int);

	//Sets the kernel and the arguments for the kernel
	cl::Kernel kernel = cl::Kernel(program, "maxTemp");
	kernel.setArg(0, buffer_A);
	kernel.setArg(1, buffer_B);
	kernel.setArg(2, cl::Local(local_size * sizeof(int)));

	//Copy to device memory
	cl::Event write_event;
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &temperature[0], NULL, &write_event);
	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

	//Runs the kernel with a thread for each element of the input with workgroups sized the same as local size
	cl::Event queue_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size), NULL, &queue_event);

	//Reads the output from GPU to CPU
	cl::Event read_event;
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &output[0], NULL, &read_event);

	//Calculate time to perform memory transfer and kernel execution
	//Time to write to memory
	long memoryWrite = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Time to read from memory
	long memoryRead = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Total time of reading and writing to memory
	long memoryExecution = memoryWrite + memoryRead;
	//Time for kernel to execute
	long kernelExecution = queue_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - queue_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Total of memory exectution and kernel execution time
	long operatingTime = memoryExecution + kernelExecution;

	//Print Max
	cout << "Max: " << (float)output.at(0) / (float)10 << endl;
	cout << "Operating time: " << operatingTime << "ns" << endl;
	totalOperatingTime += operatingTime;

}

void minTemp(cl::CommandQueue queue, cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, size_t local_size, size_t vector_size, size_t vector_elements)
{
	//Create output vector and output size
	vector<int> output(1);
	size_t output_size = output.size() * sizeof(int);

	//Sets the kernel and the arguments for the kernel
	cl::Kernel kernel = cl::Kernel(program, "minTemp");
	kernel.setArg(0, buffer_A);
	kernel.setArg(1, buffer_B);
	kernel.setArg(2, cl::Local(local_size * sizeof(int)));

	//Copy to device memory
	cl::Event write_event;
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &temperature[0], NULL, &write_event);
	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

	//Runs the kernel with a thread for each element of the input with workgroups sized the same as local size
	cl::Event queue_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size), NULL, &queue_event);

	//Reads the output from GPU to CPU
	cl::Event read_event;
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &output[0], NULL, &read_event);

	//Calculate time to perform memory transfer and kernel execution
	//Time to write to memory
	long memoryWrite = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Time to read from memory
	long memoryRead = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Total time of reading and writing to memory
	long memoryExecution = memoryWrite + memoryRead;
	//Time for kernel to execute
	long kernelExecution = queue_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - queue_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Total of memory exectution and kernel execution time
	long operatingTime = memoryExecution + kernelExecution;

	//Print Min
	cout << "Min: " << (float)output.at(0) / (float)10 << endl;
	cout << "Operating time: " << operatingTime << "ns" << endl;
	totalOperatingTime += operatingTime;

}

void standardDeviation(cl::CommandQueue queue, cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, size_t local_size, size_t vector_size, size_t vector_elements)
{
	//Create output vector and output size
	vector<unsigned int> output(1);
	size_t output_size = output.size() * sizeof(int);

	//Sets the kernel and the arguments for the kernel
	cl::Kernel kernel = cl::Kernel(program, "standardDeviation");
	kernel.setArg(0, buffer_A);
	kernel.setArg(1, buffer_B);
	kernel.setArg(2, cl::Local(local_size * sizeof(unsigned int)));
	kernel.setArg(3, mean);

	//Copy to device memory
	cl::Event write_event;
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &temperature[0], NULL, &write_event);
	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

	//Runs the kernel with a thread for each element of the input with workgroups sized the same as local size
	cl::Event queue_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size), NULL, &queue_event);

	//Reads the output from GPU to CPU
	cl::Event read_event;
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &output[0], NULL, &read_event);

	//Calculate time to perform memory transfer and kernel execution
	//Time to write to memory
	long memoryWrite = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Time to read from memory
	long memoryRead = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Total time of reading and writing to memory
	long memoryExecution = memoryWrite + memoryRead;
	//Time for kernel to execute
	long kernelExecution = queue_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - queue_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Total of memory exectution and kernel execution time
	long operatingTime = memoryExecution + kernelExecution;

	//Calculates and prints Average
	float answer = ((float)output[0] / (float)100) / (float)vector_elements;
	answer = sqrt(answer);

	cout << "Standard: " << answer << endl;
	cout << "Operating time: " << operatingTime << "ns" << endl;
	totalOperatingTime += operatingTime;

}

void median(cl::CommandQueue queue, cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, size_t local_size, size_t vector_size, size_t vector_elements)
{
	//Create output vector and output size
	vector<int> output(vector_elements);
	size_t output_size = output.size() * sizeof(int);

	//Sets the kernel and the arguments for the kernel
	cl::Kernel kernel = cl::Kernel(program, "sort");
	kernel.setArg(0, buffer_A);
	kernel.setArg(1, buffer_B);
	//kernel.setArg(2, cl::Local(local_size * sizeof(int)));

	//Copy to device memory
	cl::Event write_event;
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &temperature[0], NULL, &write_event);
	queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

	//Runs the kernel with a thread for each element of the input with workgroups sized the same as local size
	cl::Event queue_event;
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size), NULL, &queue_event);

	//Reads the output from GPU to CPU
	cl::Event read_event;
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &output[0], NULL, &read_event);

	//Calculate time to perform memory transfer and kernel execution
	//Time to write to memory
	long memoryWrite = write_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - write_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Time to read from memory
	long memoryRead = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Total time of reading and writing to memory
	long memoryExecution = memoryWrite + memoryRead;
	//Time for kernel to execute
	long kernelExecution = queue_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - queue_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	//Total of memory exectution and kernel execution time
	long operatingTime = memoryExecution + kernelExecution;

	float answer = 0;

	//Calculates and prints median
	if (vector_elements % 2 == 0) {
		answer = (((float)output[(vector_elements / 2) - 1] + output[vector_elements / 2]) / 2) / 10;
	}
	else {
		answer = ((float)output[vector_elements / 2]) / 10;
	}
	cout << "Median: " << answer << endl;

	//Calculate and print lower quartile
	answer = ((float)output[vector_elements / 4]) / 10;
	cout << "Lower Quartile: " << answer << endl;

	//Calculate and print upper quartile
	answer = ((float)output[vector_elements / 4 * 3]) / 10;
	cout << "Upper Quartile: " << answer << endl;

	cout << "Operating time: " << operatingTime << "ns" << endl;
	totalOperatingTime += operatingTime;
	cout << "Total Operating Time: " << totalOperatingTime << "ns" << endl;

}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string file = "temp_lincolnshire.txt";

	//Reads for inputs
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//Detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		cl::Program program(context, sources);

		try {
			program.build();
		}
		//Display kernel building errors
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Pass file and read the data
		readData(file);
		//Print number of records in file
		cout << "Number of records: " << temperature.size() << endl;

		//Number of elements
		size_t vector_elements = temperature.size();
		//Size in bytes
		size_t vector_size = temperature.size() * sizeof(int);
		size_t local_size = (64, 1);
		size_t padding_size = temperature.size() % local_size;

		//If vector isn't a factor of the local size, pad the vector
		if (padding_size)
		{
			std::vector<int> temp(local_size - padding_size, 1000);
			temperature.insert(temperature.end(), temp.begin(), temp.end());
		}

		//Host - output
		vector<int> C(vector_elements);

		//Device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);

		//Call each function
		maxTemp(queue, program, buffer_A, buffer_B, local_size, vector_size, vector_elements);
		minTemp(queue, program, buffer_A, buffer_B, local_size, vector_size, vector_elements);
		average(queue, program, buffer_A, buffer_B, local_size, vector_size, vector_elements);
		standardDeviation(queue, program, buffer_A, buffer_B, local_size, vector_size, vector_elements);
		median(queue, program, buffer_A, buffer_B, local_size, vector_size, vector_elements);

	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	return 0;
}

