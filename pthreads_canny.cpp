#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <pthread.h>
#include <chrono>

typedef struct {
    unsigned char* edges;
    double* suppressed;
    int height;
    int width;
    int highThreshold;
    int lowThreshold;
    int start_row;
    int end_row;
} DoubThreshData;


typedef struct {
    double* suppressed;
    double* grad_mag;
    double* grad_dir;
    int height;
    int width;
    int start_row;
    int end_row;
} NonMaxData;

typedef struct {
    double* array;
    double* out_array;
    int height;
    int width;
    struct Kernel* gauss_kernel;
    int start_row;
    int end_row;
} GuassBlurData;

typedef struct {
    double* guass_array;
    double* grad_mag;
    double* grad_dir;
    int height;
    int width;
    struct Kernel* sobel_xkernel;
    struct Kernel* sobel_ykernel;
    double* mag_max;
    int start_row;
    int end_row;
} IntGradData;

struct Kernel {
    double* data;
    int size;
};

struct Image {
    double* data;
    int height;
    int width;
};

void writeVector(unsigned char* data, int width, int height, const std::string& filename) {
    // Create a cv::Mat from the double vector
    cv::Mat mat(height, width, CV_8U, data); // Create a Mat with type double

    // Write the grayscale image to file
    if (cv::imwrite(filename, mat)) {
        std::cout << "Image successfully saved to " << filename << std::endl;
    } else {
        std::cerr << "Failed to save image to " << filename << std::endl;
    }
}

double convolve(double* image, int height, int width, struct Kernel* kernel, int row, int col) {
    int ksize = kernel->size;        // Kernel size (assuming it's square)
    int half_size = ksize / 2;       // Half size for zero padding
    
    double sum = 0.0; // Result of convolveolution at (row, col)

    for (int krow = ksize-1; krow >= 0; krow--) {
        for (int kcol = ksize-1; kcol >=0; kcol--) {
            // Calculate the image coordinates with zero padding
            int image_row = row - krow + half_size;
            int image_col = col - kcol + half_size;

            // Check for zero padding
            if (image_row < 0 || image_row >= height || 
                image_col < 0 || image_col >= width) {
                // Out of bounds, use zero for padding
                sum += 0; // Zero contribution for out-of-bounds
            } else {
                // Valid pixel, apply kernel
                sum += (image[image_row * width + image_col] * kernel->data[krow * ksize + kcol]);
            }
        }
    }

    return sum; // Return the convolveolution result
}


struct Kernel init_kernel(int size){
    struct Kernel kernel;
    kernel.size = size;
    kernel.data = (double*)malloc(size * size * sizeof(double));
    if (kernel.data == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
    return kernel;
}

void gaussian_kernel(struct Kernel* kernel, double sigma) {
    int cize = kernel->size / 2;
    double normal = 1.0 / (2.0 * M_PI * (sigma * sigma));

    // Calculate the Gaussian kernel values and store them in the 1D array
    double sum = 0.0; // For normalizing the kernel values
    for (int i = -cize; i <= cize; i++) {
        for (int j = -cize; j <= cize; j++) {
            int index = (i + cize) * kernel->size + (j + cize);  // Map 2D to 1D index
            kernel->data[index] = normal * exp(-(i * i + j * j) / (2.0 * (sigma * sigma)));
            sum += kernel->data[index];
        }
    }

    // Normalize the kernel so that the sum is 1
    for (int i = 0; i < kernel->size * kernel->size; i++) {
        kernel->data[i] /= sum;
    }
}

void sobel_kernel(struct Kernel* kernel, int axis){
    if (axis == 0){
        double values[9] = {
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0
        };
        for (int i = 0; i < kernel->size * kernel->size; i++) {
            kernel->data[i] = values[i]; // Copy values into the kernel
        } 
    } else if (axis == 1){
        double values[9] = {
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0
        };
        for (int i = 0; i < kernel->size * kernel->size; i++) {
            kernel->data[i] = values[i]; // Copy values into the kernel
        }
    }

}

void print_kernel(struct Kernel* kernel) {
    printf("Gaussian Kernel:\n");
    for (int i = 0; i < kernel->size; i++) {
        for (int j = 0; j < kernel->size; j++) {
            int index = i * kernel->size + j; // i+start_row, j+start_col 
            // printf("%d ", index);
            printf("%f ", kernel->data[index]);
        }
        printf("\n");
    }
}

void min_max(double* array, int height, int width){
    float min_val = INFINITY;
    float max_val = -INFINITY;

    for (int row = 0; row < height; row++){
        for (int col = 0; col < width; col++){
            double val = array[row * width + col];
                if (val < min_val){
                    min_val = val;
                }
                if (val > max_val){
                    max_val = val;
                }
        }
    }
    printf("min val : %f \n", min_val);
    printf("max val : %f \n", max_val);
}

struct Image load_grayscale(const std::string& filename){
    // Load the image in grayscale
    cv::Mat data = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    int height = data.rows;
    int width = data.cols;

    if (data.empty()) {
        // Return an empty Image structure if the file cannot be loaded
        std::cerr << "Failed to load image: " << filename << std::endl;
        return {nullptr, 0, 0};
    }

    // Allocate memory for a 1D array to hold the image data
    double* image_data = (double*)malloc(height * width * sizeof(double));
    if (image_data == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return {nullptr, 0, 0};;
    }

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            image_data[i * width + j] = (double)(data.at<uchar>(i, j));
        }
    }
    struct Image image;
    image.data = image_data;
    image.height = height;
    image.width = width;

    return image;
}

void* guassian_blur_thread(void* arg){
    GuassBlurData* data = (GuassBlurData*)arg;
    printf("Gauss Blur Thread ID: %lu, Start Row: %d, End Row: %d\n", pthread_self(), data->start_row, data->end_row);
    for (int out_row = data->start_row; out_row < data->end_row; out_row++) {
        for (int out_col = 0; out_col < data->width; out_col++) {
            data->out_array[out_row * data->width + out_col] = convolve(data->array, data->height, data->width, data->gauss_kernel, out_row, out_col);
        }
    }
    return NULL;
}

void guassian_blur(double* array, double* out_array, int height, int width, struct Kernel* gauss_kernel, int num_threads){
    pthread_t threads[num_threads];
    GuassBlurData thread_data[num_threads];

    int rows_per_thread = height / num_threads;
    int remaining_rows = height % num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].array = array;
        thread_data[i].out_array = out_array;
        thread_data[i].height = height;
        thread_data[i].width = width;
        thread_data[i].gauss_kernel = gauss_kernel;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i + 1) * rows_per_thread;

        // Distribute remaining rows to the last thread
        if (i == num_threads - 1) {
            thread_data[i].end_row += remaining_rows;
        }

        pthread_create(&threads[i], NULL, guassian_blur_thread, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

void* intensity_gradients_thread(void* arg){
    IntGradData* data = (IntGradData*)arg;
    printf("Intensity Grad Thread ID: %lu, Start Row: %d, End Row: %d\n", pthread_self(), data->start_row, data->end_row);
    double* y_grad = (double*)malloc(data->height * data->width * sizeof(double));
    double* x_grad = (double*)malloc(data->height * data->width * sizeof(double));

    for (int out_row = data->start_row; out_row < data->end_row; out_row++) {
        for (int out_col = 0; out_col < data->width; out_col++) {
            y_grad[out_row * data->width + out_col] = convolve(data->guass_array, data->height, data->width, data->sobel_ykernel, out_row, out_col);
            x_grad[out_row * data->width + out_col] = convolve(data->guass_array, data->height, data->width, data->sobel_xkernel, out_row, out_col);
        }
    }

    // Compute gradient magnitude and direction for the assigned rows
    double local_max = 0.0;
    for (int out_row = data->start_row; out_row < data->end_row; out_row++) {
        for (int out_col = 0; out_col < data->width; out_col++) {
            double gmval = hypot(x_grad[out_row * data->width + out_col], y_grad[out_row * data->width + out_col]);
            if (gmval > local_max) {
                local_max = gmval;
            }
            data->grad_mag[out_row * data->width + out_col] = gmval;
            data->grad_dir[out_row * data->width + out_col] = atan2(y_grad[out_row * data->width + out_col], x_grad[out_row * data->width + out_col]);
        }
    }

    // Update global max gradient value using atomic operation or reduction
    #pragma omp critical
    {
        if (local_max > *data->mag_max) {
            *data->mag_max = local_max;
        }
    }
    free(x_grad);
    free(y_grad);
    
    return NULL;
}

void intensity_gradients(double* guass_array, double* grad_mag, double* grad_dir, int height, int width, struct Kernel* sobel_xkernel, struct Kernel* sobel_ykernel, double* mag_max, int num_threads){
    pthread_t threads[num_threads];
    IntGradData thread_data[num_threads];

    int rows_per_thread = height / num_threads;
    int remaining_rows = height % num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].guass_array = guass_array;
        thread_data[i].grad_mag = grad_mag;
        thread_data[i].grad_dir = grad_dir;
        thread_data[i].height = height;
        thread_data[i].width = width;
        thread_data[i].sobel_xkernel = sobel_xkernel;
        thread_data[i].sobel_ykernel = sobel_ykernel;
        thread_data[i].mag_max = mag_max;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i + 1) * rows_per_thread;

        // Distribute remaining rows to the last thread
        if (i == num_threads - 1) {
            thread_data[i].end_row += remaining_rows;
        }

        pthread_create(&threads[i], NULL, intensity_gradients_thread, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

}

void* nonmax_suppresion_thread(void* arg) {
    NonMaxData* data = (NonMaxData*)arg;
    printf("Nonmax Supp Thread ID: %lu, Start Row: %d, End Row: %d\n", pthread_self(), data->start_row, data->end_row);
    for (int out_row = data->start_row; out_row < data->end_row; out_row++) {
        for (int out_col = 0; out_col < data->width; out_col++) {
            int index = out_row * data->width + out_col;
            int neighbor1_i;
            int neighbor1_j;
            int neighbor2_i;
            int neighbor2_j;
            double angle = data->grad_dir[index];
            if (angle > M_PI){
                angle = angle * M_PI / 180.0;
            }
            // define neighboring pixel indices based on gradient direction
            if ((0 <= angle && angle < M_PI / 8) || (15 * M_PI / 8 <= angle && angle <= 2 * M_PI)){
                neighbor1_i = out_row;
                neighbor1_j = out_col + 1;
                neighbor2_i = out_row;
                neighbor2_j = out_col - 1;
            } else if (M_PI / 8 <= angle && angle < 3 * M_PI / 8){
                neighbor1_i = out_row - 1;
                neighbor1_j = out_col + 1;
                neighbor2_i = out_row + 1;
                neighbor2_j = out_col - 1;

            } else if (3 * M_PI / 8 <= angle && angle < 5 * M_PI / 8){
                neighbor1_i = out_row - 1;
                neighbor1_j = out_col;
                neighbor2_i = out_row + 1;
                neighbor2_j = out_col;

            } else if (5 * M_PI / 8 <= angle && angle < 7 * M_PI / 8){
                neighbor1_i = out_row - 1;
                neighbor1_j = out_col - 1;
                neighbor2_i = out_row + 1;
                neighbor2_j = out_col + 1;
            } else {
                neighbor1_i = out_row - 1;
                neighbor1_j = out_col;
                neighbor2_i = out_row + 1;
                neighbor2_j = out_col;
            }
            // i is row, j is columns
            neighbor1_i = std::max(0, std::min(neighbor1_i, data->height - 1));
            neighbor1_j = std::max(0, std::min(neighbor1_j, data->width - 1));
            neighbor2_i = std::max(0, std::min(neighbor2_i, data->height - 1));
            neighbor2_j = std::max(0, std::min(neighbor2_j, data->width - 1));
            
            // compare current pixel magnitude with its neighbors along gradient di
            double current_mag = data->grad_mag[index];
            double neighbor1_mag = data->grad_mag[neighbor1_i * data->width + neighbor1_j];
            double neighbor2_mag = data->grad_mag[neighbor2_i * data->width + neighbor2_j];
    
            // perform supression
            if ((current_mag >= neighbor1_mag) && (current_mag >= neighbor2_mag)){
                data->suppressed[index] = current_mag;
            } else{
                data->suppressed[index] = 0;
            }
        }
    }
    return NULL;
}

void nonmax_suppresion(double* suppressed, double* grad_mag, double* grad_dir, int height, int width, int num_threads){
    pthread_t threads[num_threads];
    NonMaxData thread_data[num_threads];

    int rows_per_thread = height / num_threads;
    int remaining_rows = height % num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].suppressed = suppressed;
        thread_data[i].grad_mag = grad_mag;
        thread_data[i].grad_dir = grad_dir;
        thread_data[i].height = height;
        thread_data[i].width = width;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i + 1) * rows_per_thread;

        // Distribute remaining rows to the last thread
        if (i == num_threads - 1) {
            thread_data[i].end_row += remaining_rows;
        }

        pthread_create(&threads[i], NULL, nonmax_suppresion_thread, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
}

void* double_threshold_thread(void* arg) {
    DoubThreshData* data = (DoubThreshData*)arg;
    unsigned char weak = 25;
    unsigned char strong = 255;
    printf("Double Thresh Thread ID: %lu, Start Row: %d, End Row: %d\n", pthread_self(), data->start_row, data->end_row);

    for (int out_row = data->start_row; out_row < data->end_row; out_row++) {
        for (int out_col = 0; out_col < data->width; out_col++) {
            int index = out_row * data->width + out_col;
            data->edges[index] = 0;
            if (data->suppressed[index] >= data->highThreshold) {
                data->edges[index] = strong;
            } else if (data->suppressed[index] >= data->lowThreshold) {
                data->edges[index] = weak;
            }
        }
    }
    return NULL;
}

void double_threshold(unsigned char* edges, double* suppressed, int height, int width, int highThreshold, int lowThreshold, int num_threads){
    pthread_t threads[num_threads];
    DoubThreshData thread_data[num_threads];

    int rows_per_thread = height / num_threads;
    int remaining_rows = height % num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].edges = edges;
        thread_data[i].suppressed = suppressed;
        thread_data[i].height = height;
        thread_data[i].width = width;
        thread_data[i].highThreshold = highThreshold;
        thread_data[i].lowThreshold = lowThreshold;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i + 1) * rows_per_thread;

        // Distribute remaining rows to the last thread
        if (i == num_threads - 1) {
            thread_data[i].end_row += remaining_rows;
        }

        pthread_create(&threads[i], NULL, double_threshold_thread, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

void edge_tracking_hysteresis(unsigned char* edges, int height, int width){
    // edge tracking by hysteresis
    unsigned char weak = 25;
    unsigned char strong = 255;
    for (int out_row = 0; out_row < height; out_row++){
        for (int out_col = 0; out_col < width; out_col++){
            if (edges[out_row * width + out_col] == weak){
                if (
                    (edges[(out_row+1) * width + out_col-1] == strong) || (edges[(out_row+1) * width + out_col] == strong) ||
                    (edges[(out_row+1) * width + out_col+1] == strong) || (edges[out_row * width + out_col-1] == strong) ||
                    (edges[out_row * width + out_col+1] == strong) || (edges[(out_row-1) * width + out_col-1] == strong) ||
                    (edges[(out_row-1) * width + out_col] == strong) || (edges[(out_row-1) * width + out_col+1] == strong)
                ){
                    edges[out_row * width + out_col] = strong;
                } else {
                    edges[out_row * width + out_col] = 0;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
     // Check for correct usage
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <number of threads>" << std::endl;
        return EXIT_FAILURE;
    }
    int num_threads = atoi(argv[2]);

    auto start = std::chrono::high_resolution_clock::now();
    double highThresholdRatio = 0.08;
    double lowThresholdRatio = 0.02;

    struct Kernel gauss_kernel = init_kernel(3);
    gaussian_kernel(&gauss_kernel, 1.0);
    struct Kernel sobel_ykernel = init_kernel(3);
    sobel_kernel(&sobel_ykernel, 0);
    struct Kernel sobel_xkernel = init_kernel(3);
    sobel_kernel(&sobel_xkernel, 1);

    struct Image image = load_grayscale(argv[1]);

    double* noise_reduced = (double*)malloc(image.height * image.width * sizeof(double));
    guassian_blur(image.data, noise_reduced, image.height, image.width, &gauss_kernel, num_threads);
    free(image.data);

    double mag_max = -INFINITY;
    double* grad_mag = (double*)malloc(image.height * image.width * sizeof(double));
    double* grad_dir = (double*)malloc(image.height * image.width * sizeof(double));
    intensity_gradients(noise_reduced, grad_mag, grad_dir, image.height, image.width, &sobel_xkernel, &sobel_ykernel, &mag_max, num_threads);
    free(noise_reduced);
    free(gauss_kernel.data);
    free(sobel_ykernel.data);
    free(sobel_xkernel.data);

    double* suppressed = (double*)malloc(image.height * image.width * sizeof(double));
    nonmax_suppresion(suppressed, grad_mag, grad_dir, image.height, image.width, num_threads);
    free(grad_mag);
    free(grad_dir);


    double highThreshold = mag_max * highThresholdRatio;
    double lowThreshold = highThreshold * lowThresholdRatio;

    
    unsigned char* edges = (unsigned char*)malloc(image.height * image.width * sizeof(unsigned char));
    double_threshold(edges, suppressed, image.height, image.width, highThreshold, lowThreshold, num_threads);
    free(suppressed);

    edge_tracking_hysteresis(edges, image.height, image.width);
    writeVector(edges, image.width, image.height, "./outputs/pthread_canny.png");

    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    free(edges);

    return EXIT_SUCCESS;
}
// g++ pthreads_canny.cpp -o pthreads_canny `pkg-config --cflags --libs opencv4` -pthread
// ./pthreads_canny castle.png
