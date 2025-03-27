#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <fstream>
#include <iomanip>

using namespace cv;

// Helper function to create a Hann window
static cv::Mat createHannWindow(int win_length, bool periodic) {
    cv::Mat window(1, win_length, CV_32F);
    float* ptr = window.ptr<float>();

    // 计算归一化因子
    int N = periodic ? win_length : win_length - 1;

    for (int i = 0; i < win_length; ++i) {
        ptr[i] = 0.5 * (1 - std::cos(2 * CV_PI * i / N));
    }

    return window;
}

// Helper function to pad center
static cv::Mat padCenter(const cv::Mat& window, int size) {
    if (window.cols >= size) {
        return window.clone();
    }

    int pad_left = (size - window.cols) / 2;
    int pad_right = size - window.cols - pad_left;

    cv::Mat padded_window;
    cv::copyMakeBorder(window, padded_window, 0, 0, pad_left, pad_right, cv::BORDER_CONSTANT, 0);
    return padded_window;
}

// Helper function to frame the signal
static cv::Mat frameSignal(const cv::Mat& y, int frame_length, int hop_length) {
    if (y.cols < frame_length) {
        throw std::runtime_error("Frame length exceeds signal length");
    }

    int n_frames = 1 + (y.cols - frame_length) / hop_length;
    cv::Mat frames(n_frames, frame_length, y.type());

    for (int i = 0; i < n_frames; ++i) {
        int start = i * hop_length;
        int end = start + frame_length;
        y.colRange(start, end).copyTo(frames.row(i));
    }

    return frames;
}


// Main STFT function
cv::Mat stft(const cv::Mat& y, int n_fft = 2048, int hop_length = -1,
    int win_length = -1, const std::string& window = "hann",
    bool center = true, int dtype = CV_32F,
    const std::string& pad_mode = "constant") {
    // Validate input
    if (y.empty()) {
        throw std::runtime_error("Input signal is empty");
    }
    if (y.rows != 1) {
        throw std::runtime_error("Only single-channel signals are supported");
    }

    // Set default win_length
    if (win_length <= 0) {
        win_length = n_fft;
    }

    // Set default hop_length
    if (hop_length <= 0) {
        hop_length = win_length / 4;
    }

    // Create window
    cv::Mat fft_window;
    if (window == "hann") {
        fft_window = createHannWindow(win_length, true);
    }
    else {
        throw std::runtime_error("Only Hann window is currently implemented");
    }

    // Pad the window to n_fft size
    fft_window = padCenter(fft_window, n_fft);

    // Pad the time series if centered
    cv::Mat y_padded;
    int start = 0;
    int extra = 0;
    cv::Mat y_frames_pre, y_frames_post;

    if (center) {
        if (pad_mode != "constant") {
            throw std::runtime_error("Only constant padding is currently supported");
        }

        if (n_fft > y.cols) {
            std::cerr << "Warning: n_fft=" << n_fft
                << " is too large for input signal of length=" << y.cols << std::endl;
        }

        // Calculate padding
        int padding = n_fft / 2;
        cv::copyMakeBorder(y, y_padded, 0, 0, padding, padding, cv::BORDER_CONSTANT, 0);

        // Calculate how many frames depend on left padding
        int start_k = static_cast<int>(std::ceil(n_fft / 2.0 / hop_length));

        // Calculate the first frame that depends on extra right-padding
        int tail_k = (y.cols + n_fft / 2 - n_fft) / hop_length + 1;

        if (tail_k <= start_k) {
            // Simple case - just pad both sides
            cv::copyMakeBorder(y, y_padded, 0, 0, padding, padding, cv::BORDER_CONSTANT, 0);
        }
        else {
            // Complex case - handle head and tail separately
            start = start_k * hop_length - n_fft / 2;

            // Handle head frames
            int head_end = (start_k - 1) * hop_length - n_fft / 2 + n_fft + 1;
            head_end = std::min(head_end, y.cols);

            cv::Mat y_head = y.colRange(0, head_end);
            cv::copyMakeBorder(y_head, y_head, 0, 0, padding, 0, cv::BORDER_CONSTANT, 0);
            y_frames_pre = frameSignal(y_head, n_fft, hop_length);
            y_frames_pre = y_frames_pre.rowRange(0, start_k);

            extra = y_frames_pre.rows;

            // Handle tail frames
            int tail_start = tail_k * hop_length - n_fft / 2;
            if (tail_start < y.cols) {
                cv::Mat y_tail = y.colRange(tail_start, y.cols);
                cv::copyMakeBorder(y_tail, y_tail, 0, 0, 0, padding, cv::BORDER_CONSTANT, 0);
                y_frames_post = frameSignal(y_tail, n_fft, hop_length);
                extra += y_frames_post.rows;
            }
        }
    }
    else {
        if (n_fft > y.cols) {
            throw std::runtime_error("n_fft is too large for uncentered analysis");
        }
        y_padded = y.clone();
    }

    // Frame the signal
    cv::Mat y_frames = frameSignal(y.colRange(start, y.cols), n_fft, hop_length);
    std::cout << y_frames_pre << std::endl;

    // Determine output shape
    int n_frames = y_frames.rows + extra;
    int n_bins = 1 + n_fft / 2;

    // Create output matrix (2-channel for complex numbers)
    cv::Mat stft_matrix(n_bins, n_frames, CV_MAKETYPE(dtype, 2));
    stft_matrix.setTo(0);

    // Apply window and compute FFT
    cv::Mat windowed_frame;
    cv::Mat complex_frame;

    // 通用窗函数扩展方法
    auto expand_window = [](const cv::Mat& window, int rows) {
        cv::Mat expanded;
        cv::repeat(window.reshape(1, 1), rows, 1, expanded);
        return expanded;
        };

    if (y_frames_pre.rows > 0) {
        cv::Mat windowed_pre;
        cv::Mat window_expanded = expand_window(fft_window, y_frames_pre.rows);
        cv::multiply(y_frames_pre, window_expanded, windowed_pre);

        cv::Mat complex_pre;
        cv::dft(windowed_pre, complex_pre, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);

        // 转置使得每列是一个帧的FFT结果
        cv::Mat complex_pre_t;
        cv::transpose(complex_pre, complex_pre_t);

        // 只取前n_bins个频点
        complex_pre_t.rowRange(0, n_bins).copyTo(stft_matrix.colRange(0, y_frames_pre.rows));
    }

    // 将窗函数转换为与帧矩阵相同的维度

    // 批量处理主帧
    if (y_frames.rows > 0) {
        cv::Mat windowed_main;
        cv::Mat window_expanded = expand_window(fft_window, y_frames.rows);
        cv::multiply(y_frames, window_expanded, windowed_main);

        cv::Mat complex_main;
        cv::dft(windowed_main, complex_main, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);

        // 转置并复制到结果矩阵
        cv::Mat complex_main_t;
        cv::transpose(complex_main, complex_main_t);

        int start_col = y_frames_pre.rows;
        complex_main_t.rowRange(0, n_bins).copyTo(
            stft_matrix.colRange(start_col, start_col + y_frames.rows));
    }

    // 批量处理后处理帧
    if (y_frames_post.rows > 0) {
        cv::Mat windowed_post;
        cv::Mat window_expanded = expand_window(fft_window, y_frames_post.rows);
        cv::multiply(y_frames_post, window_expanded, windowed_post);

        cv::Mat complex_post;
        cv::dft(windowed_post, complex_post, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);

        cv::Mat complex_post_t;
        cv::transpose(complex_post, complex_post_t);

        int start_col = y_frames_pre.rows + y_frames.rows;
        complex_post_t.rowRange(0, n_bins).copyTo(
            stft_matrix.colRange(start_col, start_col + y_frames_post.rows));
    }

    printComplexMat(stft_matrix);

    return stft_matrix;
}

// Helper function for overlap-add operation
static void overlapAdd(cv::Mat& dst, const cv::Mat& src, int hop_length) {
    if (dst.empty()) {
        src.copyTo(dst);
        return;
    }

    int n_samples = src.cols;
    int n_frames = src.rows;

    for (int i = 0; i < n_frames; ++i) {
        int start = i * hop_length;
        int end = start + n_samples;

        if (end > dst.cols) {
            end = dst.cols;
            n_samples = end - start;
        }

        cv::Mat dst_roi = dst(cv::Rect(start, 0, n_samples, 1));
        cv::Mat src_roi = src.row(i)(cv::Rect(0, 0, n_samples, 1));
        dst_roi += src_roi;
    }
}

template<typename T>
static void expandToNFFT_Impl(const Mat& inputCol, Mat& outputCol, int nFFT) {
    int originalRows = inputCol.rows;
    outputCol.create(nFFT, 1, inputCol.type());

    // 复制前originalRows行
    inputCol.rowRange(0, originalRows).copyTo(outputCol.rowRange(0, originalRows));

    // 填充共轭对称部分
    for (int i = originalRows; i < nFFT; ++i) {
        int originalIdx = nFFT - i;
        if (originalIdx >= originalRows) originalIdx = 0;  // 避免越界

        // 获取共轭复数（根据类型处理）
        Vec<T, 2> val = inputCol.at<Vec<T, 2>>(originalIdx, 0);
        outputCol.at<Vec<T, 2>>(i, 0) = Vec<T, 2>(val[0], -val[1]);
    }
}

Mat irfft(const Mat& stftMatrix, int nFFT) {
    int numCols = stftMatrix.cols;
    int depth = stftMatrix.depth();
    Mat output;

    // 根据输入类型创建输出矩阵
    if (depth == CV_32F) {
        output = Mat(nFFT, numCols, CV_32F);
    }
    else if (depth == CV_64F) {
        output = Mat(nFFT, numCols, CV_64F);
    }
    else {
        CV_Error(Error::StsUnsupportedFormat, "Only CV_32F or CV_64F are supported");
    }

    for (int col = 0; col < numCols; ++col) {
        Mat currentCol = stftMatrix.col(col);

        // 根据类型扩展频谱
        Mat extendedCol;
        if (depth == CV_32F) {
            expandToNFFT_Impl<float>(currentCol, extendedCol, nFFT);
        }
        else {
            expandToNFFT_Impl<double>(currentCol, extendedCol, nFFT);
        }

        // 执行逆DFT（输出实数）
        Mat inverseTransform;
        idft(extendedCol, inverseTransform, DFT_SCALE | DFT_REAL_OUTPUT, 0);

        // 复制到输出列（保持类型一致）
        inverseTransform.col(0).copyTo(output.col(col));
    }

    return output;
}


// Window sum square calculation
static cv::Mat windowSumSquare(const std::string& window, int n_frames, int win_length,
    int n_fft, int hop_length, int dtype) {
    // Create window
    cv::Mat window_vals;
    if (window == "hann") {
        window_vals = createHannWindow(win_length, true);
    }
    else {
        throw std::runtime_error("Only Hann window is currently implemented");
    }

    // Pad window to n_fft
    if (win_length < n_fft) {
        int pad_left = (n_fft - win_length) / 2;
        int pad_right = n_fft - win_length - pad_left;
        cv::copyMakeBorder(window_vals, window_vals, 0, 0, pad_left, pad_right, cv::BORDER_CONSTANT, 0);
    }

    // Square the window
    cv::Mat win_sq;
    cv::multiply(window_vals, window_vals, win_sq);

    // Calculate the sum
    cv::Mat ifft_window_sum = cv::Mat::zeros(1, n_fft + hop_length * (n_frames - 1), CV_32F);

    for (int i = 0; i < n_frames; ++i) {
        int start = i * hop_length;
        int end = start + n_fft;
        if (end > ifft_window_sum.cols) {
            end = ifft_window_sum.cols;
        }

        cv::Mat roi = ifft_window_sum(cv::Rect(start, 0, end - start, 1));
        roi += win_sq(cv::Rect(0, 0, end - start, 1));
    }

    return ifft_window_sum;
}

// Main ISTFT function
cv::Mat istft(const cv::Mat& stft_matrix, int hop_length = -1, int win_length = -1,
    int n_fft = -1, const std::string& window = "hann", bool center = true,
    int dtype = CV_32F, int length = -1) {
    // Validate input
    if (stft_matrix.empty()) {
        throw std::runtime_error("Input STFT matrix is empty");
    }

    // Infer n_fft from input shape if not provided
    if (n_fft <= 0) {
        n_fft = 2 * (stft_matrix.rows - 1);
    }

    // Set default win_length
    if (win_length <= 0) {
        win_length = n_fft;
    }

    // Set default hop_length
    if (hop_length <= 0) {
        hop_length = win_length / 4;
    }

    // Create window
    cv::Mat ifft_window;
    if (window == "hann") {
        ifft_window = createHannWindow(win_length, true);
    }
    else {
        throw std::runtime_error("Only Hann window is currently implemented");
    }

    // Pad window to n_fft
    ifft_window = padCenter(ifft_window, n_fft);

    // Determine number of frames
    int n_frames = stft_matrix.cols;
    if (length > 0) {
        int padded_length = center ? length + 2 * (n_fft / 2) : length;
        n_frames = std::min(n_frames, static_cast<int>(std::ceil(padded_length / static_cast<float>(hop_length))));
    }

    // Calculate expected signal length
    int expected_signal_len = n_fft + hop_length * (n_frames - 1);
    if (length > 0) {
        expected_signal_len = length;
    }
    else if (center) {
        expected_signal_len -= 2 * (n_fft / 2);
    }

    // Create output matrix
    cv::Mat y = cv::Mat::zeros(1, expected_signal_len, dtype);

    // For complex-to-real conversion
    cv::Mat stft_real, stft_imag;
    std::vector<cv::Mat> channels;
    cv::split(stft_matrix, channels);
    stft_real = channels[0];
    stft_imag = channels[1];

    // Calculate start frame and offset if centered
    int start_frame = 0;
    int offset = 0;

    // 通用窗函数扩展方法
    auto expand_window = [](const cv::Mat& window, int rows) {
        cv::Mat expanded;
        cv::repeat(window.reshape(1, 1), rows, 1, expanded);
        return expanded;
        };

    if (center) {
        start_frame = static_cast<int>(std::ceil((n_fft / 2) / static_cast<float>(hop_length)));

        // Process head block
        if (start_frame > 0) {
            // 1. 提取头部STFT帧
            cv::Mat head_stft = stft_matrix(cv::Rect(0, 0, start_frame, stft_matrix.rows));
            //printComplexMat(head_stft);
            cv::Mat ytmp = irfft(head_stft, n_fft);
            //printComplexMat(ytmp);

            cv::Mat window_expanded = expand_window(ifft_window, ytmp.cols);
            cv::transpose(ytmp, ytmp);
            cv::multiply(ytmp, window_expanded, ytmp);
            //printComplexMat(ytmp);

            // Create head buffer
            int head_buffer_len = n_fft + hop_length * (start_frame - 1);
            cv::Mat head_buffer = cv::Mat::zeros(1, head_buffer_len, dtype);

            // Overlap-add
            overlapAdd(head_buffer, ytmp, hop_length);
            //printComplexMat(head_buffer);

            // Copy to output
            // 处理输出缓冲区的复制
            if (y.cols < head_buffer.cols - n_fft / 2) {
                // 如果y比head_buffer小，取全部能容纳的部分
                head_buffer(cv::Rect(n_fft / 2, 0, y.cols, 1)).copyTo(y);
            }
            else {
                // 否则只复制head_buffer的有效部分到y
                int copy_len = head_buffer.cols - n_fft / 2;
                head_buffer(cv::Rect(n_fft / 2, 0, copy_len, 1))
                    .copyTo(y(cv::Rect(0, 0, copy_len, 1)));
            }

            offset = start_frame * hop_length - n_fft / 2;
        }
        else
        {
            start_frame = 0;
            offset = 0;
        }
    }

    // Process remaining frames in blocks
    int n_columns = 256; // Process 256 columns at a time (adjust based on memory constraints)

    for (int bl_s = start_frame; bl_s < n_frames; bl_s += n_columns) {
        int bl_t = std::min(bl_s + n_columns, n_frames);
        int block_width = bl_t - bl_s;

        // 1. 获取当前块 (单边频谱 9xN)
        cv::Mat current_stft = stft_matrix(cv::Rect(bl_s, 0, block_width, stft_matrix.rows));
        cv::Mat ytmp = irfft(current_stft, n_fft);

        // 5. 应用窗函数 (需确保ifft_window是16x1)
        cv::Mat window_expanded = expand_window(ifft_window, ytmp.cols);
        cv::transpose(ytmp, ytmp);
        cv::multiply(ytmp, window_expanded, ytmp);
        //printComplexMat(ytmp);

        // Overlap-add
        int start_pos = (bl_s - start_frame) * hop_length + offset;
        cv::Mat y_roi = y(cv::Rect(start_pos, 0, y.cols - start_pos, 1));
        overlapAdd(y_roi, ytmp, hop_length);
    }

    // Normalize by sum of squared window
    cv::Mat ifft_window_sum = windowSumSquare(window, n_frames, win_length, n_fft, hop_length, dtype);
    //printComplexMat(ifft_window_sum);

    int start_pos = center ? n_fft / 2 : 0;
    if (ifft_window_sum.cols > start_pos) {
        cv::Mat valid_window_sum = ifft_window_sum(cv::Rect(start_pos, 0,
            std::min(ifft_window_sum.cols - start_pos, y.cols), 1));

        // Avoid division by zero
        float threshold = 1e-10; // Small value threshold
        for (int i = 0; i < valid_window_sum.cols; ++i) {
            float val = valid_window_sum.at<float>(0, i);
            if (val > threshold) {
                if (dtype == CV_32F) {
                    y.at<float>(0, i) /= static_cast<float>(val);
                }
                else {
                    y.at<float>(0, i) /= val;
                }
            }
        }
    }

    return y;
}


