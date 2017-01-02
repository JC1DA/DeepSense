#pragma OPENCL EXTENSION cl_khr_fp16 : enable

static inline int getIndexFrom3D(int d1, int d2, int d3, int i1, int i2, int i3) {
	return i1 * (d2 * d3) + i2 * d3 + i3;
}

static inline int getIndexFrom4D(int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4) {
	return i1 * (d2 * d3 * d4) + i2 * (d3 * d4) + i3 * d4 + i4;
}

kernel void convertFloatToHalf(
    global const float *input,
    global half *output) {
    int idx = get_global_id(0);
    vstore_half(input[idx], 0, &output[idx]);
}

kernel void convertHalfToFloat(
    global const half *input,
    global float *output) {
    int idx = get_global_id(0);
    //output[idx] = convert_float(input[idx]);
    output[idx] = (float)input[idx];
}

__kernel void conv_kernel_half(
    __global const half *input,
    const int input_w,
    const int input_h,
    const int input_c,
    __global const half *conv_weight,
    __global const half *bias,
    const int conv_w,
    const int conv_h,
    const int conv_c,
    const int conv_n,
    const int stride_w,
    const int stride_h,
    const int pad_left,
    const int pad_right,
    const int pad_top,
    const int pad_bot,
    __global half *output,
    const int output_w,
    const int output_h,
    const int output_c) {
    int x,y,z,n,i,j;

    int threadId_x = get_global_id(0);
    int threadId_y = get_global_id(1);
    int threadId_z = get_global_id(2);

    int useBase3 = (input_c % 3 == 0) ? 1 : 0;

    for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
        for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
            for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
                half result = 0.0f;
                for(y = 0 ; y < conv_h ; y++) {
                    int global_input_y = j * stride_h - pad_top + y;
                    for(x = 0 ; x < conv_w ; x++) {
                        int global_input_x = i * stride_w - pad_left + x;
                        if(global_input_x >= 0 && global_input_y >= 0 && global_input_x < input_w && global_input_y < input_h) {
                            if(useBase3 == 1) {
                                for(z = 0 ; z < conv_c ; z += 3) {
                                    int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, x, z);
                                    int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);
                                    
                                    half2 tmp_input = vload2(0, &input[global_input_index]);
                                    half2 tmp_weight = vload2(0, &conv_weight[global_filter_index]);
                                    result += dot(tmp_input, tmp_weight);
                                    
                                    result += input[global_input_index + 2] * conv_weight[global_filter_index + 2];
                                }
                            } else {
                                for(z = 0 ; z < conv_c ; z += 16) {
                                    int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, x, z);
                                    int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);

                                    half16 tmp_input = vload16(0, &input[global_input_index]);
                                    half16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);

                                    result += dot(tmp_input.s0123, tmp_weight.s0123);
                                    result += dot(tmp_input.s4567, tmp_weight.s4567);
                                    result += dot(tmp_input.s89ab, tmp_weight.s89ab);
                                    result += dot(tmp_input.scdef, tmp_weight.scdef);
                                }
                            }
                        }
                    }
                }

                result += bias[n];

                output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result;
            }
        }
    }
}

__kernel void conv_kernel_float(
	__global const float *input,
    const int input_w,
    const int input_h,
    const int input_c,
    __global const float *conv_weight,
    __global const float *bias,
    const int conv_w,
    const int conv_h,
    const int conv_c,
    const int conv_n,
    const int stride_w,
    const int stride_h,
    const int pad_left,
    const int pad_right,
    const int pad_top,
    const int pad_bot,
    __global float *output,
    const int output_w,
    const int output_h,
    const int output_c) {
	int x,y,z,n,i,j;

    int threadId_x = get_global_id(0);
    int threadId_y = get_global_id(1);
    int threadId_z = get_global_id(2);

    int useBase3 = (input_c % 3 == 0) ? 1 : 0;

    for(n = threadId_z ; n < output_c ; n += get_global_size(2)) {
    	for(j = threadId_y ; j < output_h ; j += get_global_size(1)) {
    		for(i = threadId_x ; i < output_w ; i += get_global_size(0)) {
    			float result = 0.0f;
    			for(y = 0 ; y < conv_h ; y++) {
    				int global_input_y = j * stride_h - pad_top + y;
    				for(x = 0 ; x < conv_w ; x++) {
    					int global_input_x = i * stride_w - pad_left + x;
    					if(global_input_x >= 0 && global_input_y >= 0 && global_input_x < input_w && global_input_y < input_h) {
    						if(useBase3 == 1) {
    							for(z = 0 ; z < conv_c ; z += 3) {
    								int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, x, z);
    								int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);
    								
    								float2 tmp_input = vload2(0, &input[global_input_index]);
	                                float2 tmp_weight = vload2(0, &conv_weight[global_filter_index]);
	                                result += dot(tmp_input, tmp_weight);
	                                
	                                result += input[global_input_index + 2] * conv_weight[global_filter_index + 2];
    							}
    						} else {
    							for(z = 0 ; z < conv_c ; z += 16) {
	                                int global_filter_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, n, y, x, z);
    								int global_input_index = getIndexFrom3D(input_h, input_w, input_c, global_input_y, global_input_x, z);

	                                float16 tmp_input = vload16(0, &input[global_input_index]);
	                                float16 tmp_weight = vload16(0, &conv_weight[global_filter_index]);

	                                result += dot(tmp_input.s0123, tmp_weight.s0123);
	                                result += dot(tmp_input.s4567, tmp_weight.s4567);
	                                result += dot(tmp_input.s89ab, tmp_weight.s89ab);
	                                result += dot(tmp_input.scdef, tmp_weight.scdef);
	                            }
    						}
    					}
    				}
    			}

    			result += bias[n];

	            output[getIndexFrom3D(output_h, output_w, output_c, j, i, n)] = result;
    		}
    	}
    }
}

__kernel void conv_fc_kernel_half(
    __global const half *input,
    const int input_w,
    const int input_h,
    const int input_c,
    __global const half *conv_weight,
    __global const half *bias,
    const int conv_w,
    const int conv_h,
    const int conv_c,
    const int conv_n,
    const int stride_w,
    const int stride_h,
    const int pad_left,
    const int pad_right,
    const int pad_top,
    const int pad_bot,
    __global half *output,
    const int output_w,
    const int output_h,
    const int output_c
) {
	for(int threadId_x = get_global_id(0) ; threadId_x < output_c ; threadId_x += get_global_size(0)) {
		int i;
	    int weight_start_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, threadId_x, 0, 0, 0);
	    float result = 0.0f;

	    int remaining = conv_w * conv_h * conv_c;
	    i = 0;
	    while(remaining > 0 && remaining / 16 > 0) {
	    	half16 tmp_input = vload16(0, &input[i]);
	    	half16 tmp_weight = vload16(0, &conv_weight[weight_start_index + i]);

	    	result += dot(tmp_input.s0123, tmp_weight.s0123);
	        result += dot(tmp_input.s4567, tmp_weight.s4567);
	        result += dot(tmp_input.s89ab, tmp_weight.s89ab);
	        result += dot(tmp_input.scdef, tmp_weight.scdef);

	        remaining -= 16;
	        i += 16;
	    }

	    while(remaining > 0 && remaining / 4 > 0) {
	    	half4 tmp_input = vload4(0, &input[i]);
	    	half4 tmp_weight = vload4(0, &conv_weight[weight_start_index + i]);

	    	result += dot(tmp_input, tmp_weight);

	    	remaining -= 4;
	    	i += 4;
	    }

	    while(remaining > 0) {
	    	result += input[i] * conv_weight[weight_start_index + i];

	    	remaining--;
	    	i++;
	    }

	    result += bias[threadId_x];

	    output[threadId_x] = result;
	    //vstore_half(result, 0, &output[threadId_x]);
	}
}

__kernel void conv_fc_kernel_float(
    __global const float *input,
    const int input_w,
    const int input_h,
    const int input_c,
    __global const float *conv_weight,
    __global const float *bias,
    const int conv_w,
    const int conv_h,
    const int conv_c,
    const int conv_n,
    const int stride_w,
    const int stride_h,
    const int pad_left,
    const int pad_right,
    const int pad_top,
    const int pad_bot,
    __global float *output,
    const int output_w,
    const int output_h,
    const int output_c
) {
	for(int threadId_x = get_global_id(0) ; threadId_x < output_c ; threadId_x += get_global_size(0)) {
		int weight_start_index = getIndexFrom4D(conv_n, conv_h, conv_w, conv_c, threadId_x, 0, 0, 0);
	    float result = 0.0f;

	    int remaining = conv_w * conv_h * conv_c;
	    int i = 0;
	    while(remaining > 0 && remaining / 16 > 0) {
	    	float16 tmp_input = vload16(0, &input[i]);
	    	float16 tmp_weight = vload16(0, &conv_weight[weight_start_index + i]);

	    	result += dot(tmp_input.s0123, tmp_weight.s0123);
	        result += dot(tmp_input.s4567, tmp_weight.s4567);
	        result += dot(tmp_input.s89ab, tmp_weight.s89ab);
	        result += dot(tmp_input.scdef, tmp_weight.scdef);

	        remaining -= 16;
	        i += 16;
	    }

	    while(remaining > 0 && remaining / 4 > 0) {
	    	float4 tmp_input = vload4(0, &input[i]);
	    	float4 tmp_weight = vload4(0, &conv_weight[weight_start_index + i]);

	    	result += dot(tmp_input, tmp_weight);

	    	remaining -= 4;
	    	i += 4;
	    }

	    while(remaining > 0) {
	    	result += input[i] * conv_weight[weight_start_index + i];

	    	remaining--;
	    	i++;
	    }

	    result += bias[threadId_x];

	    output[threadId_x] = result;
	}
}

kernel void fully_connected_kernel_half(
    global const half *input_frame,
    const int input_w,
    const int input_h,
    const int input_d,
    global const half *layer_W,
    global const half *layer_bias,
    global half *output_frame,
    const int output_size
) {
    int thrIdx = get_global_id(0);
    int maxThreads = get_global_size(0);

    for(int n = thrIdx; n < output_size ; n += maxThreads) {
        float result = 0.0f;
        
        int input_idx = 0;
        int filter_idx = n * input_h * input_w * input_d;

        int idx_remaining = input_h * input_w * input_d;

        while(idx_remaining >= 4) {
            half4 tmp1 = vload4(0, &input_frame[input_idx]);
            half4 tmp2 = vload4(0, &layer_W[filter_idx]);
            result += dot(tmp1,tmp2);

            input_idx += 4;
            filter_idx += 4;
            idx_remaining -= 4;
        }

        while(idx_remaining >= 2) {
            half2 tmp1 = vload2(0, &input_frame[input_idx]);
            half2 tmp2 = vload2(0, &layer_W[filter_idx]);
            result += dot(tmp1,tmp2);

            input_idx += 2;
            filter_idx += 2;
            idx_remaining -= 2;
        }

        while(idx_remaining > 0) {
            half tmp1 = input_frame[input_idx];
            half tmp2 = layer_W[filter_idx];
            result += tmp1 * tmp2;

            idx_remaining -= 1;
        }

        result += layer_bias[n];

        output_frame[n] = result;
    }
}

kernel void fully_connected_kernel_float(
    global const float *input_frame,
    const int input_w,
    const int input_h,
    const int input_d,
    global const float *layer_W,
    global const float *layer_bias,
    global float *output_frame,
    const int output_size
) {
    int thrIdx = get_global_id(0);
    int maxThreads = get_global_size(0);

    for(int n = thrIdx; n < output_size ; n += maxThreads) {
        float result = 0.0f;
        
        int input_idx = 0;
        int filter_idx = n * input_h * input_w * input_d;

        int idx_remaining = input_h * input_w * input_d;

        while(idx_remaining >= 4) {
            float4 tmp1 = vload4(0, &input_frame[input_idx]);
            float4 tmp2 = vload4(0, &layer_W[filter_idx]);
            result += dot(tmp1,tmp2);

            input_idx += 4;
            filter_idx += 4;
            idx_remaining -= 4;
        }

        while(idx_remaining > 0) {
            float tmp1 = input_frame[input_idx];
            float tmp2 = layer_W[filter_idx];
            result += tmp1 * tmp2;

            idx_remaining -= 1;
        }

        result += layer_bias[n];

        output_frame[n] = result;
    }
}

__kernel void maxpool_kernel_half(
    __global const half *input_frame,
    const int input_w,
    const int input_h,
    const int input_d,
    const int size,
    const int stride_1,
    const int stride_2,
    const int pad_1,
    const int pad_2,
    const int pad_3,
    const int pad_4,
    __global half *output_frame,
    const int output_w,
    const int output_h,
    const int output_d) {

    int thrId_i = get_global_id(0);
    int thrId_j = get_global_id(1);
    int thrId_k = get_global_id(2);

    int max_i = get_global_size(0);
    int max_j = get_global_size(1);
    int max_k = get_global_size(2);

    int i,j,k;
    int x,y;

    for(i = thrId_i ; i < output_w ; i += max_i) {
        for(j = thrId_j ; j < output_h ; j += max_j) {
            for(k = thrId_k ; k < output_d ; k += max_k) {
                half max = -9999.9f;
                for(x = 0 ; x < size ; x++) {
                    for(y = 0 ; y < size ; y++) {
                        int x_ = i * stride_1 + x - pad_1;
                        int y_ = j * stride_2 + y - pad_3;
                        int valid = (x_ >= 0 && x_ < input_w && y_ >= 0 && y_ < input_h);
                        //float val = (valid != 0) ? input_frame[getIndexFrom3D(input_h, input_w, input_d, y_, x_, k)] : -999999.9f;
                        half val = (valid != 0) ? input_frame[getIndexFrom3D(input_h, input_w, input_d, y_, x_, k)] : 0.0f;
                        max   = (val > max) ? val   : max;
                    }
                }
                output_frame[getIndexFrom3D(output_h, output_w, output_d, j, i, k)] = max;
                //vstore_half(max, 0, &output_frame[getIndexFrom3D(output_h, output_w, output_d, j, i, k)]);
            }
        }
    }
}

__kernel void maxpool_kernel_float(
    __global const float *input_frame,
    const int input_w,
    const int input_h,
    const int input_d,
    const int size,
    const int stride_1,
    const int stride_2,
    const int pad_1,
    const int pad_2,
    const int pad_3,
    const int pad_4,
    __global float *output_frame,
    const int output_w,
    const int output_h,
    const int output_c) {

    int thrId_i = get_global_id(0);
    int thrId_j = get_global_id(1);
    int thrId_k = get_global_id(2);

    int max_i = get_global_size(0);
    int max_j = get_global_size(1);
    int max_k = get_global_size(2);

    int i,j,k;
    int x,y;

    for(i = thrId_i ; i < output_w ; i += max_i) {
        for(j = thrId_j ; j < output_h ; j += max_j) {
            for(k = thrId_k ; k < output_c ; k += max_k) {
                float max = -9999.9f;
                for(x = 0 ; x < size ; x++) {
                    for(y = 0 ; y < size ; y++) {
                        int x_ = i * stride_1 + x - pad_1;
                        int y_ = j * stride_2 + y - pad_3;
                        int valid = (x_ >= 0 && x_ < input_w && y_ >= 0 && y_ < input_h);
                        //float val = (valid != 0) ? input_frame[getIndexFrom3D(input_h, input_w, input_d, y_, x_, k)] : -999999.9f;
                        float val = (valid != 0) ? input_frame[getIndexFrom3D(input_h, input_w, input_d, y_, x_, k)] : 0.0f;
                        max   = (val > max) ? val   : max;
                    }
                }
                output_frame[getIndexFrom3D(output_h, output_w, output_c, j, i, k)] = max;
            }
        }
    }
}

__kernel void cross_channels_lrn_kernel_half(
    __global const half *in, //[h x w x c]
    const int channels,
    const int height,
    const int width,
    const int k,
    const int size,
    const float alpha_over_size,
    const float beta,
    __global half *out) {

    half beta_half = 0.0f;
    vstore_half(beta, 0, &beta_half);

    for(int w = get_global_id(0) ; w < width ; w += get_global_size(0)) {
        for(int h = get_global_id(1) ; h < height ; h += get_global_size(1)) {
            int offset = (h * width + w) * channels;
            int head = 0;
            int pre_pad = (size - 1) / 2;
            int post_pad = size - pre_pad - 1;
            half accum_scale = 0;

            while (head < post_pad) {
                half data = in[offset + head];
                accum_scale += data * data;
                head++;
            }

            while (head < size) {
                half data = in[offset + head];
                accum_scale += data * data;
                half scale = k + accum_scale * alpha_over_size;
                out[offset + head - post_pad] = in[offset + head - post_pad] * pow(scale, -beta_half);
                head++;
            }

            while (head < channels) {
                half data = in[offset + head];
                accum_scale += data * data;
                data = in[offset + head - size];
                accum_scale -= data * data;
                half scale = k + accum_scale * alpha_over_size;
                out[offset + head - post_pad] = in[offset + head - post_pad] * pow(scale, -beta_half);
                head++;
            }

            while (head < channels + post_pad) {
                half data = in[offset + head - size];
                accum_scale -= data * data;
                half scale = k + accum_scale * alpha_over_size;
                out[offset + head - post_pad] = in[offset + head - post_pad] * pow(scale, -beta_half);
                head++;
            }
        }
    }
}

__kernel void cross_channels_lrn_kernel_float(
    __global const float *input, //[h x w x c]
    const int channels,
    const int height,
    const int width,
    const int k,
    const int size,
    const float alpha_over_size,
    const float beta,
    __global float *output) {

    for(int w = get_global_id(0) ; w < width ; w += get_global_size(0)) {
        for(int h = get_global_id(1) ; h < height ; h += get_global_size(1)) {
            int offset = getIndexFrom3D(height, width, channels, h, w, 0);
            int head = 0;
            int pre_pad = (size - 1) / 2;
            int post_pad = size - pre_pad - 1;
            float accum_scale = 0;

            const __global float *in = input + offset;
            __global float *out = output + offset;

            while (head < post_pad) {
                float data = in[head];
                accum_scale += data * data;
                head++;
            }

            while (head < size) {
                float data = in[head];
                accum_scale += data * data;
                float scale = k + accum_scale * alpha_over_size;
                out[head - post_pad] = in[head - post_pad] * pow(scale, -beta);
                head++;
            }

            while (head < channels) {
                float data = in[head];
                accum_scale += data * data;
                data = in[head - size];
                accum_scale -= data * data;
                float scale = k + accum_scale * alpha_over_size;
                out[head - post_pad] = in[head - post_pad] * pow(scale, -beta);
                head++;
            }

            while (head < channels + post_pad) {
                float data = in[head - size];
                accum_scale -= data * data;
                float scale = k + accum_scale * alpha_over_size;
                out[ head - post_pad] = in[head - post_pad] * pow(scale, -beta);
                head++;
            }
        }
    }
}

__kernel void activation_kernel_half(
	__global half *data,
	const int activation) {
	half result = data[get_global_id(0)];

	switch(activation) {
        case 0:
            //no activation
            break;
        case 1:
            //RAMP
            result = result * (result > 0) + 0.1 * result;
            break;
        case 2:
            //LOGISTIC
            result = 1.0 / (1.0 + exp(-result));
            break;
        case 3:
            //LEAKY
            result = (result > 0) ? result : 0.1 * result;
            break;
        case 4:
            //LINEAR
            break;
        case 5:
            //RELU
            result = (result > 0) ? result : 0.0f;
            break;
    }

    data[get_global_id(0)] = result;
}

__kernel void activation_kernel_float(
	__global float *data,
	const int activation) {
	float result = data[get_global_id(0)];

	switch(activation) {
        case 0:
            //no activation
            break;
        case 1:
            //RAMP
            result = result * (result > 0) + 0.1 * result;
            break;
        case 2:
            //LOGISTIC
            result = 1.0 / (1.0 + exp(-result));
            break;
        case 3:
            //LEAKY
            result = (result > 0) ? result : 0.1 * result;
            break;
        case 4:
            //LINEAR
            break;
        case 5:
            //RELU
            result = (result > 0) ? result : 0.0f;
            break;
    }

    data[get_global_id(0)] = result;
}