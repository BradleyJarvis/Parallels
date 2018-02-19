__kernel void avg(__global const int* temperature, __global int* output, __local int* scratch) {
	//Get IDs and size
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//Cache all values to local memory
	scratch[lid] = temperature[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	//Loop through values and adds them together
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Cache to output
	if (!lid) {
		atom_add(&output[0], scratch[lid]);

	}
}

__kernel void maxTemp(__global const int* temperature, __global int* output, __local int* scratch) {
	//Get IDs and size
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//Cache all values to local memory
	scratch[lid] = temperature[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	//Loop through values and gets highest value
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			if (scratch[lid] < scratch[lid + i]) {
				(scratch[lid] = scratch[lid + i]);
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Cache to output
	if (!lid) {
		atom_max(&output[0], scratch[lid]);

	}
}

__kernel void minTemp(__global const int* temperature, __global int* output, __local int* scratch) {
	//Get IDs and size
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//Cache all values to local memory
	scratch[lid] = temperature[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	//Loop through values and gets lowest value
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			if (scratch[lid] > scratch[lid + i]) {
				(scratch[lid] = scratch[lid + i]);
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Cache to output
	if (!lid) {
		atom_min(&output[0], scratch[lid]);

	}
}

__kernel void standardDeviation(__global const int* temperature, __global uint* output, __local uint* scratch, int mean) {
	//Get IDs and size
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//Cache all values to local memory and get sum of squared differences
	scratch[lid] = (temperature[id] - mean) * (temperature[id] - mean);

	barrier(CLK_LOCAL_MEM_FENCE);

	//Loop through values and adds them together
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Cache to output
	if (!lid) {
		atom_add(&output[0], scratch[lid]);

	}
}

__kernel void sort(__global const int* temperature, __global int*output)
{
	//Get IDs and size
	int id = get_global_id(0);
	int N = get_global_size(0);
	float temp = temperature[id];

	//Compute position of temperature in output
	int pos = 0;
	for (int i = 0; i < N; i++)
	{
		float temp2 = temperature[i];
		bool smaller = (temp2 < temp) || (temp2 == temp && i < id);
		pos += (smaller) ? 1 : 0;
	}

	//Cache to output
	output[pos] = temp;
}
