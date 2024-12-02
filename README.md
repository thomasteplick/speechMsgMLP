<h3>Speech Recognition using a Multilayer Perceptron with Back-propagation and Spectrogram Input</h3>
<p>
This is a web application written in Go that make use of the html/template package to dynamically create the web page.
Start the web server at bin\speechmlp.exe and connect to it from your web browser at http://127.0.0.1:8080/speechMsgMLP.
The program reads in audio WAV files and creates spectrograms from them.  Spectrograms are 3-dimensional plots of the time-frequency
content of the audio.  The spectral power at a particular frequency and time is shown as a grayscale color, with black having the 
greatest power and white having the least.  Short-time Fourier transforms (STFT) are used in 32 ms time intervals with 50% overlap
between FFTs.  The sections are multiplied with various window types to minimize spectral leakage due to the Gibbs phenomenon.
Time domain plots of the audio waveform can be displayed as well as the spectrogram.  The user can enter audio of not more than
two seconds by selecting the <i>New Speech</i> radio button, as well as some previously created WAV file.  Upon selecting the
<i>Play Speech</i> radio button, the user will hear the contents of the audio WAV file.  This program requires the <b>fmedia</b>
programto be in the PATH environmental variable.
</p>
<p>
The FFT size is 256.  The sampling rate is 8,000 Hz which produces a Nyquist critical frequency of 4,000 Hz.
The checkbox <i>Words Only</i> can be checked to eliminate the spectral noise for the intervals when no
audio is present.  This occurs at the beginning and end of the audio and in-between words.  The <i>Word Window</i>
select dropdown determines the sample window to separate the beginning and ending of the words in the audio.
This is used to eliminate the noise that is present in the spectrogram.  The window types used to reduce
the spectral leakage from the high sidelobes of the rectangular window are Hamming, Welch, Hanning and 
The grayscale consists of five colors.  As stated above, black RGB(0,0,0), has the greatest power in the PSD bin.
PSD bins with power more than 10dB down from the maximum bin power are white RGB(255,255,255).
</p>
<p>
The spectrograms are used as input to the Multilayer Perceptron (MLP) Neural Network.  The spectrograms are
serialized and submitted to the input layer of the MLP.  Backpropagation is used to train the weights in the
network.  The <i>Train</i> page allows the speech and frequency domain parameters to be chosen. Upon sumitting
the form in the HTML, the audio WAV files are converted to spectrograms and submitted to the MLP NN.  Next the
<i>Test</i> page will submit a test WAV file.  You can enter a new one by selecting the <i>New Message</i> radio button,
otherwise the saved test WAV file will be played.  You can plot the time domain or the spectrogram of the test WAV file.
The <i>Words Only</i> checkbox will only display the spectrogram of the actual speech excluding the noise.  You will
hear the vocabulary word that is displayed.
</p>
<p>
The <i>Vocabulary</i> allows you to enter vocabulary words.  You select the name of the word and the folder in which
it is stored.  You can also display the time domain and spectrogram of any vocabulary word that is saved.  The 
<i>Spectrogram</i> page allows you to view time domain or spectrogram plots. You can play a current WAV file
or enter a new one.  This page is similar to the <i>Test</i> page.
</p>


<h4>Learning Curve (MSE vs Epoch)</h4>
![speechMessageMLP_2](https://github.com/user-attachments/assets/bc98c8a2-81ba-4ae8-a50e-70e027c69eb5)
<h4>Time Domain (Amplitude/sec)</h4>
![speechMessageMLP_3](https://github.com/user-attachments/assets/16160073-f08a-434d-9989-690fa998af40)
<h4>Spectrogram, words only (Hz/sec)</h4>
![speechMessageMLP_4](https://github.com/user-attachments/assets/7fe60f49-6221-4ac7-9006-3a81b27555c3)
<h4>Spectrogram, includes noise, (Hz/sec)</h4>
![speechMessageMLP_5](https://github.com/user-attachments/assets/8a5670cd-604b-4ca6-95bb-18f6bc159638)




                                      
