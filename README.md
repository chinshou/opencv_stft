# opencv_stft
STFT and ISTFT librosa algorithm implemented with opencv

# Usage

channel_stft = stft(single_channel, n_fft, hop_length, n_fft, "hann", true, CV_32F);

Mat reconstructed = istft(channel_stft, hop_length, n_fft, n_fft, "hann", true, CV_32F, signal_len);
